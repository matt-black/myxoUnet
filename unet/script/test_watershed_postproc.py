#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testing the watershed post-processing technique

@author: matt
"""

# %% imports
# python
import os, sys
import json
import argparse

import torch
from torchvision.transforms import CenterCrop

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

sys.path.insert(1, os.path.abspath('..'))
from train import load_checkpoint
from data import MaskDataset

from util import process_image

# %% script arguments
parser = argparse.ArgumentParser(description="UNet training evaluation script")

parser.add_argument("-f", "--folder", type=str,
                    default="../../expts/2021-08-15_dsc", 
                    help="path to training output folder")
parser.add_argument("-d", "--data", type=str, 
                    default="../supp/test",
                    help="path to evaluation data folder")
args = parser.parse_args()

# %% load
# training arguments
with open(os.path.join(args.folder, "args.json")) as args_json:
    train_args = json.load(args_json)
    try:
        train_args["data_statnorm"]
    except:
        train_args["data_statnorm"] = False
    train_args = argparse.Namespace(**train_args)
    
# network
net, _ = load_checkpoint(os.path.join(args.folder, "model.pth"),
                         train_args)

# load test dataset
if args.data is not None:
    data_path = args.data
else:
    data_path = train_args.data
data = MaskDataset(data_path, "test", train_args.num_classes,
                   transform=None, nplicates=1, 
                   stat_norm=train_args.data_statnorm)

# %% watershed
img, msk = data.__getitem__(1)
cc = CenterCrop([train_args.crop_size+train_args.input_pad, 
                 train_args.crop_size+train_args.input_pad])
cc2 = CenterCrop([train_args.crop_size, train_args.crop_size])
pred = process_image(cc(img), net)
cell_mask = torch.argmax(pred, dim=1).squeeze(0).detach().numpy() == 1
pr_cell = pred[0,1,:,:]
pr_edge = pred[0,2,:,:]
ws_strt = (pred[0,2,:,:] - pred[0,1,:,:]).detach().numpy()
markers, _ = ndi.label (cell_mask)
ws_res = watershed (cc2(img).squeeze(0).detach().numpy(), markers, 
                    connectivity=np.ones((3,3)), mask=cell_mask, 
                    watershed_line=True)
plt.imshow(ws_res)