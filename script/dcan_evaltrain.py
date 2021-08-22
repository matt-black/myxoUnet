#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% imports
# python
import os, sys
import csv
import json
import argparse

from torch.nn import functional as F
from torchvision.transforms import CenterCrop

# science libs
import pandas as pd
import matplotlib.pyplot as plt

# my stuff
sys.path.insert(1, os.path.abspath('..'))
from train_dcan import load_checkpoint
from data import DCANDataset

from util import overlap_tile, process_image, truefalse_posneg_stats

# %% script arguments
parser = argparse.ArgumentParser(description="DCAN training evaluation script")

parser.add_argument("-f", "--folder", type=str,
                    default="./dcan_2021-08-22", 
                    help="path to training output folder")
parser.add_argument("-d", "--data", type=str, 
                    default="./supp/dcankc",
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
data = DCANDataset(data_path, "test",
                   transform=None,
                   stat_norm=train_args.data_statnorm)

# %% Katie's image
img, msk, cntr = data.__getitem__(0)
crop_size = train_args.crop_size
cc = CenterCrop([crop_size, crop_size])
img = cc(img)
msk = cc(msk)

m0, mc, _ = net(img.unsqueeze(0))
p0 = F.softmax(m0, dim=1)
pc = F.softmax(mc, dim=1)
# plt.imshow(pre[0,1,:,:])

img_disp = CenterCrop([train_args.crop_size, train_args.crop_size])(img)

plt.figure()
plt.imshow(img_disp[0,:,:,].numpy(), cmap='gray')
plt.figure()
plt.imshow(pc[0,1,:,:].detach().numpy(), cmap='jet')
plt.show()
