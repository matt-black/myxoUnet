#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% imports
# python
import os
import csv
import json
import argparse
# pytorch

# science libs
import pandas as pd
import matplotlib.pyplot as plt
# my stuff
from train import load_checkpoint
from data import MaskDataset

from util import overlap_tile

# %% script arguments
parser = argparse.ArgumentParser(description="UNet training evaluation script")

parser.add_argument("-f", "--folder", type=str,
                    default="./2021-07-19_ce", 
                    help="path to training output folder")
parser.add_argument("-d", "--data", type=str, 
                    default=None,
                    help="path to evaluation data folder")
args = parser.parse_args()

# %% load
# training arguments
with open(os.path.join(args.folder, "args.json")) as args_json:
    train_args = json.load(args_json)
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
data = MaskDataset(data_path, "test", train_args.num_classes,
                   transform=None, nplicates=1)

# %% overlap tiling
img, msk = data.__getitem__(0)
pred = overlap_tile(img.unsqueeze(0), net, 
                    train_args.crop_size, train_args.input_pad)
