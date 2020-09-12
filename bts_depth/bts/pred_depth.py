from ycb_loader import PoseDataset
from pytorch.bts_modular import BTS

ds = PoseDataset(mode="test") # mode = train / test

img, depth, label, calibrate_params = ds[0]

print(len(ds), img.shape, depth.shape, label, calibrate_params)

parser = None
depth_model = BTS(parser)
depth_model.cuda()


targets = []
preds = []

for img, depth, label, calibrate_params in ds:
    img = torch.tensor(img).cuda()
    depth = torch.tensor(depth)
    pred_depth = depth_model.forward(img)
    
    targets.append(depth.cpu())
    preds.append(pred_depth.cpu())
    
targets = torch.stack(targets)
preds = torch.stack(preds)

rmse = (targets - preds).pow(2).sum().sqrt()
print("RMSE: ", rmse)
    
