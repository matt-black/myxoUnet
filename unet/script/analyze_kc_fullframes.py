#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run net on Katie's data
@author: matt
"""
# %% imports
# python
import os, sys
import json
import argparse

import torch

# science libs
import numpy as np
from scipy.io import loadmat, savemat

# my stuff
sys.path.insert(1, os.path.abspath('..'))
from train import load_checkpoint
from util import overlap_tile

# %% script arguments
parser = argparse.ArgumentParser(description="UNet for full frames of Katie's data")

parser.add_argument("-f", "--folder", type=str,
                    default="/Users/kcopenhagen/Documents/unet/2021-08-30_wce/", 
                    help="path to training output folder")
parser.add_argument("-m", "--mat", type=str, 
                    default="/Users/kcopenhagen/Documents/MATLAB/gitstuff/myxoUnet/supp/cellLabeler/labaled_cells_08_06_2021.mat",
                    help="matlab file")
args = parser.parse_args()

# %%
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

I = np.zeros((8, 768, 1024))
P = np.zeros((8, 768, 1024))
data = loadmat (args.mat)["dataset1"]
for ix in range(1,9):
    # load image from mat file 
    # don't know why you need the [0,0]
    img = data["l{:d}".format(ix)][0,0].astype(int)
    
    # convert to pytorch
    img = torch.from_numpy(img)
    
    # normalize image
    # if train_args.data_statnorm:
    #     subval = img.float().mean()
    #     denom = img.float().std()
    # else:
    #     subval = img.min().float()
    #     denom = img.max().float() - subval
    # img = (img.float() - subval) / denom
    
    pred = overlap_tile(img, net, train_args.crop_size, 
                        train_args.input_pad//2, "stat")
    # save results
    I[ix-1,:,:] = img.numpy()
    P[ix-1,:,:] = pred.detach().numpy()
    print("Done with image {:d}".format(ix))
savemat("analyze_kc_fullframes.mat", {"I" : I, "P": P})
