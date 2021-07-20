#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% imports
# python
import os
import csv
import json
import argparse

import torchmetrics.functional as tmF

# science libs
import pandas as pd
import matplotlib.pyplot as plt
# my stuff
from train import load_checkpoint
from data import MaskDataset

from util import overlap_tile, truefalse_posneg_stats

# %% script arguments
parser = argparse.ArgumentParser(description="UNet training evaluation script")

parser.add_argument("-f", "--folder", type=str,
                    default="./expts/stat_norm/2021-07-20_jreg_adam_200e", 
                    help="path to training output folder")
parser.add_argument("-d", "--data", type=str, 
                    default=None,
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
    
# losses
train_loss = pd.read_csv(os.path.join(args.folder, "losses.csv"))

# network    
net, _ = load_checkpoint(os.path.join(args.folder, "model.pth"),
                         train_args)

# load test dataset
if args.data is not None:
    data_path = args.data
else:
    data_path = train_args.data
data = MaskDataset(data_path, "train", train_args.num_classes,
                   transform=None, nplicates=1, 
                   stat_norm=train_args.data_statnorm)

# %% overlap tiling
img, msk = data.__getitem__(1)
pred = overlap_tile(img.unsqueeze(1), net, 
                    train_args.crop_size, train_args.input_pad)
# (tp,fp), (tn,fn) = truefalse_posneg_stats(msk, pred, 2)
# ss = tmF.stat_scores(pred, msk, num_classes=2, multiclass=False)
# ss2 = tmF.iou(pred, msk, num_classes=1)
plt.imshow(pred==1)