#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% imports
# python
import os, sys
import csv
import json
import argparse

import torchmetrics.functional as tmF
from torchvision.transforms import CenterCrop

# science libs
import pandas as pd
import matplotlib.pyplot as plt

# my stuff
sys.path.insert(1, os.path.abspath('..'))
from train import load_checkpoint
from data import MaskDataset

from util import overlap_tile, process_image, truefalse_posneg_stats

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
    
# losses
train_loss = pd.read_csv(os.path.join(args.folder, "losses.csv"))

plt.plot(train_loss["epoch"], train_loss["train"],
          train_loss["epoch"], train_loss["test"])

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

# %% Katie's image
# img, msk = data.__getitem__(2)
# crop_size = train_args.crop_size + train_args.input_pad
# cc = CenterCrop([crop_size, crop_size])
# img = cc(img)
# msk = cc(msk)

# pre = process_image(img.unsqueeze(0), net).detach().numpy()
# # plt.imshow(pre[0,1,:,:])

# img_disp = CenterCrop([train_args.crop_size, train_args.crop_size])(img)

# plt.figure()
# plt.imshow(img_disp[0,:,:,].numpy(), cmap='gray')
# plt.figure()
# plt.imshow(pre[0,1,:,:], cmap='jet')
# plt.show()
# %% overlap tiling
img, msk = data.__getitem__(0)
pred = overlap_tile(img.unsqueeze(1), net, 
                    train_args.crop_size, train_args.input_pad//2)
# (tp,fp), (tn,fn) = truefalse_posneg_stats(msk, pred, 2)
# ss = tmF.stat_scores(pred, msk, num_classes=2, multiclass=False)
# ss2 = tmF.iou(pred, msk, num_classes=1)
plt.imshow(pred==1)