#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCAN testing/experimentation script
@author: matt
"""
# %% imports
# python
import os, sys
import argparse

from torchvision.transforms import CenterCrop

# my stuff
sys.path.insert(1, os.path.abspath('../dcan'))
from dcan import DCAN
from data import DCANDataset
from util import convTranspose2dOutputSize, conv2dOutputSize

# %% script arguments
parser = argparse.ArgumentParser(description="DCAN testing script")

parser.add_argument("-d", "--data", type=str, 
                    default="../supp/dcankc",
                    help="path to evaluation data folder")
parser.add_argument("-c", "--crop-size", type=int,
                    default=263)
args = parser.parse_args()

# %%
data = DCANDataset(args.data, "test", transform=None, stat_norm=True)

net = DCAN(in_channels=1, 
           depth=5, 
           wf=4, 
           kernel_size=3, 
           batch_norm=True,
           up_mode='upconv',
           out_dim=256)

img, msk, cntr = data.__getitem__(0)
cc = CenterCrop([args.crop_size, args.crop_size])
mask, cntr, c123 = net(cc(img).unsqueeze(0))
