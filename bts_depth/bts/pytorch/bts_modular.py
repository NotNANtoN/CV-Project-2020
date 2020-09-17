from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from .bts_dataloader import *
from .bts import BtsModel

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def define_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
    parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                        default='densenet121_bts')
    #parser.add_argument('--data_path', type=str, help='path to the data', required=True)
    #parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
    parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='bts_depth/bts/pytorch/models/bts_nyu_v2_pytorch_densenet121/model')
    parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
    parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
    return parser

    
def read_args(parser):
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    return args

# model_dir = os.path.dirname(args.checkpoint_path)
# sys.path.append(model_dir)



class WrappedModel(nn.Module):
	def __init__(self, args):
		super(WrappedModel, self).__init__()
		self.module = BtsModel(params=args) # that I actually define.
	def forward(self, *args, **kwargs):
		return self.module(*args, **kwargs)


class BTS(torch.nn.Module):
    def __init__(self, parser,args=None, dataparallel=True):
        super().__init__()
        # set args
        if args is None:
            parser = define_parser(parser)
            args = read_args(parser)
        else:
            args = args
        args.mode = 'test'
        # add model dir to path
        model_dir = os.path.dirname(args.checkpoint_path)
        sys.path.append(model_dir)
        # init model
        #self.model = BtsModel(params=args)
        self.model = WrappedModel(args)
        if dataparallel:
            self.model = torch.nn.DataParallel(self.model)
        
        # load weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if dataparallel:
            checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(self.device))
        else:
            checkpoint = torch.load(args.checkpoint_path, map_location={"cuda" : "cpu"})
            
        self.model.load_state_dict(checkpoint['model'])
        # to cuda if possible
        self.model.to(self.device)
        self.model.eval()
        
        factor = 1 # 518.8579
        self.focal = torch.tensor(factor).to(self.device).float()  # For NYU
        
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        #if x.shape[-1] == 3:
        #    x = x.permute(2, 0, 1)
        if x.ndim == 3:
            x = x.view(1, *list(x.shape))
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.to(self.device).float()
        #factor = np.random.random() * 5000
        #print("Factor: ", factor)
        factor = 1
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = self.model(x, factor, self.device)
        return depth_est
        



