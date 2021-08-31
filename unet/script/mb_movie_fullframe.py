#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% imports
# python
import os, sys
import csv
import json
import argparse

import torch
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F

# science libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# my stuff
sys.path.insert(1, os.path.abspath(os.path.join('..')))
from train import load_checkpoint
from data import MaskDataset
from util import overlap_tile
sys.path.insert(1, os.path.abspath(os.path.join('..','..','supp')))
import vk4extract.vk4extract as vk4e

# %% script arguments
parser = argparse.ArgumentParser(description="UNet training evaluation script")

parser.add_argument("-f", "--folder", type=str,
                    default="../../../expts/2021-08-30_wce", 
                    help="path to training output folder")
parser.add_argument("-d", "--data", type=str,
                    default="/media/matt/UNTITLED/work/h2o/data/videos/100x/2021-08-28_frzE/frzE_highP/view1")
args = parser.parse_args()

# %% load checkpoint
# load training args
with open(os.path.join(args.folder, "args.json")) as args_json:
    train_args = json.load(args_json)
    train_args = argparse.Namespace(**train_args)
# load trained network
net, _ = load_checkpoint(os.path.join(args.folder, "model.pth"), 
                         train_args)

# if we computed global stats, need to figure out normalization
if train_args.data_global_stats:
    dset = MaskDataset(os.path.join("..", train_args.data), "train",
                       train_args.num_classes, None, True)
    normalize = dset.normalize

# %% convenience functions

def read_laser(fpath, as_torch=False):
    with open(fpath, 'rb') as f:
        off = vk4e.extract_offsets(f)
        lsr_dict = vk4e.extract_img_data(off, 'light', f)
        lsr_data = lsr_dict['data']
        hgt = lsr_dict['height']
        wid = lsr_dict['width']
        lsr = np.reshape(lsr_data, (hgt, wid))
    return lsr

def make_frm_path(frno):
    return os.path.join(args.data, "img", "frame{:06d}.vk4".format(frno))
    
def to_torch(I):
    return torch.from_numpy(I.astype('float32')).unsqueeze(0) / 65535

# %% look at example sequence

frames = []
fig = plt.figure(figsize=(12,9))
# main loop
for i in range(83):
    # read laser image in, save as uint8 type
    lsr = to_torch(read_laser(make_frm_path(i+1)))
    lsr_uint8 = (lsr * 255).type(torch.ByteTensor).squeeze(0)
    if train_args.data_global_stats:
        lsr = normalize(lsr)
    else:
        lsr = (lsr - lsr.mean()) / lsr.std()
    msk = overlap_tile(lsr, net, 
                       train_args.crop_size, 
                       train_args.input_pad//2)
    # draw segmentation mask
    I = draw_segmentation_masks(lsr_uint8.repeat(3,1,1),
                                torch.stack([msk==i for i in range(4)], dim=0),
                                alpha=0.4, 
                                colors=["black","red","blue","green"])
    frames.append([plt.imshow(F.to_pil_image(I), aspect='auto', animated=True)])

anim = animation.ArtistAnimation(fig, frames, interval=500, blit=True)
anim.save('example_seg_movie.mp4')
plt.show()