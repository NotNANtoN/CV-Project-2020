import os
from argparse import ArgumentParser
from warnings import warn
from collections import defaultdict

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import transforms
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')

from pytorch.bts import silog_loss
from pytorch.bts_modular import define_parser, read_args, BTS
from ycb_loader import PoseDataset


class LitBTS(LightningModule):
    def __init__(self, args):
        super().__init__()
        
        self.lr = args.lr
        self.batch_size = args.batch_size
        # Create model
        self.model = BTS(None, args=args)
        self.model.train()
        self.silog_criterion = silog_loss(variance_focus=args.variance_focus)
        self.m_logs = defaultdict(list)

    def forward(self, x):
        depth_est = self.model(x)
        return depth_est.squeeze()
    
    def update_m_logs(self, dict_):
        for key in dict_:
            self.m_logs[key].append(dict_[key].item())

    def training_step(self, batch, batch_idx):
        img, depth_gt, label, calibrate_params, folder = batch
        depth_gt = depth_gt.float() / 10000
        pred_depth = self(img) / 10
        mask = (depth_gt > 0.1).type_as(img).bool()
        loss = self.silog_criterion.forward(pred_depth, depth_gt, mask)
        
        tensorboard_logs = {'train_loss': loss}
        progress_bar_metrics = tensorboard_logs
        self.update_m_logs(tensorboard_logs)
        return {
            'loss': loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def validation_step(self, batch, batch_idx):
        img, depth_gt, label, calibrate_params, folder = batch
        depth_gt = depth_gt.float() / 10000
        with torch.no_grad():
            pred_depth = self(img) / 10
        mask = (depth_gt > 0.1).type_as(img).bool()
        loss = self.silog_criterion.forward(pred_depth, depth_gt, mask)
        mae = (depth_gt - pred_depth).abs().mean()
        logs = {'val_loss': loss, 'val_mae': mae}
        self.update_m_logs(logs)
        return {'val_loss': loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        weight_decay = 1e-2
        optimizer = torch.optim.AdamW([{'params': self.model.model.module.encoder.parameters(), 'weight_decay': weight_decay},
                                   {'params': self.model.model.module.decoder.parameters(), 'weight_decay': 0}],
                                  lr=self.lr)
        return optimizer

    def train_dataloader(self):
        ds = PoseDataset(mode="train") # mode = train / test
        dataloader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self):
        ds = PoseDataset(mode="test") # mode = train / test
        dataloader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        return dataloader


def extend_parser(parser):
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=10)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--lr',             type=float, help='initial learning rate', default=1e-6)
    parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
    return parser
    
def cli_main():
    # args
    parser = define_parser()
    parser = extend_parser(parser)
    args = read_args(parser)
    args.checkpoint_path = "pytorch/models/bts_nyu_v2_pytorch_densenet121/model"

    # model
    model = LitBTS(args)

    # training
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       min_delta=0.00,
       patience=5,
       verbose=False,
       mode='min'
    )
    from pytorch_lightning.loggers import MLFlowLogger
    mlf_logger = MLFlowLogger(
        experiment_name="default",
        tracking_uri="file:./ml-runs"
    )
    trainer = Trainer(gpus=1,
                      early_stop_callback=early_stop_callback,
                      val_percent_check=0.005,
                      val_check_interval=1 / 1000,
                      precision=16,
                      auto_scale_batch_size=None,
                      logger=mlf_logger)
    trainer.fit(model)
    from time import gmtime, strftime
    timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    torch.save(model.m_logs, f"logs{timestr}.pt")
    del model.m_logs
    torch.save(model.model.state_dict(), f"finetuned{timestr}.pt")

if __name__ == '__main__':  # pragma: no cover
    cli_main()
