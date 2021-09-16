# -*- coding: utf-8 -*-
"""Process a single movie

@author: mb46
"""
# %% imports
# python
import os, sys
import json
import argparse
import random
import fnmatch

import torch
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F

# science libs
import numpy as np
from pandas import read_csv
from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# my stuff
sys.path.insert(1, os.path.abspath(os.path.join('..')))
from train import load_checkpoint
from data import MaskDataset
from util import overlap_tile

sys.path.insert(1, os.path.abspath(os.path.join('..','..','supp')))
import vk4extract.vk4extract as vk4e

def main(**kwargs):
    # process input args
    args = argparse.Namespace(**kwargs)
    
    # use cuda?
    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    if not args.quiet:
        if use_cuda:
            print("using cuda")
        else:
            print("using cpu")
    
    # setup output folder
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    else:  # folder exists, add random int so we dont get overlap
        args.output = args.output + "_" + \
            "{:d}".format(random.randint(1,1000))
        os.mkdir(args.output)
        if not args.quiet:
            print("specified output folder exists, new one is {:s}".format(
                    args.output))
     
    # load training arguments
    with open(os.path.join(args.checkpoint, "args.json")) as aj:
        train_args = json.load(aj)
        train_args = argparse.Namespace(**train_args)
    
    # setup trained network
    net, _, _ = load_checkpoint(os.path.join(args.checkpoint, "model.pth"), 
                                train_args)
    net = net.to(device)
    if not args.quiet:
        print("made net, transferred to device")
    
    # if we computed global stats, need to figure out normalization scheme
    if train_args.data_global_stats:
        if args.training_data is None:
            raise Exception("must specify --training_data if you trained on global stats")
        dset = MaskDataset(args.training_data, "train", 
                           train_args.num_classes, None, True)
        normalize = dset.normalize
        if not args.quiet:
            print("computed normalization transform")

    # formulate kwargs dict for watershed
    ws_kwargs = {"gauss_sigma" : args.gauss_sigma,
                 "rho_mask" : args.rho_mask,
                 "rho_seed" : args.rho_seed,
                 "seed_neig_power" : args.seed_neig_power}
    
    if args.data_format == "ktl":  # ktlapp style VK4 + frameTimeZ csv
        # read in frameTimeZ csv
        frTZ = read_csv(os.path.join(args.data, "frameTimeZ.csv"))
        max_fr = max(frTZ["frame"])
        if not args.quiet:
            print("loaded frameTimeZ, will process {:d} frames".format(max_fr))
        
        for fr in range(1, max_fr):  # NOTE: skips actual last frame
            # read in image
            fpath = os.path.join(args.data, "img", 
                                 "frame{:06d}.vk4".format(fr))
            lsr = _to_torch(read_vk4image(fpath, 'light'))
            # normalize to [0,1] then do stats
            lsr = (lsr.float() - lsr.min().float()) / \
                (lsr.max().float()-lsr.min().float())
            if train_args.data_global_stats:
                lsr = normalize(lsr)
            else:
                lsr = (lsr - lsr.mean()) / lsr.std()
            # do full-frame prediction w/ overlap tile method
            lsr = lsr.to(device)
            if args.output_format == "watershed":
                ws = overlap_tile(lsr, net,
                                  crop_size=train_args.crop_size,
                                  pad_size=train_args.input_pad//2,
                                  output="watershed",
                                  **ws_kwargs)
                # move to numpy and force onto cpu
                ws = ws.detach().cpu().numpy()
                # save to *.mat format
                savemat(os.path.join(args.output, "frame{:06d}.mat".format(fr)),
                        {"W" : ws})
            else:
                cd, nd = overlap_tile(lsr, net,
                                      crop_size=train_args.crop_size,
                                      pad_size=train_args.input_pad//2,
                                      output="distance")
                # move to numpy, force onto cpu
                cd = cd.detach().cpu().numpy()
                nd = nd.detach().cpu().numpy()
                # save to *.mat format
                savemat(os.path.join(args.output, "frame{:06d}.mat".format(fR)),
                        {"cell_dist" : cd, "neig_dist" : nd})
            if not args.quiet:
                print("saved frame {:d}".format(fr))
            
    elif args.data_format == "kc":  # katie's format with *.bin
        # read in times txt
        dfpath = os.path.join(args.data,"Laser")
        max_fr = len(fnmatch.filter(os.listdir(dfpath),'*.bin'))
        if not args.quiet:
            print("loaded times, will process {:d} frames".format(max_fr))
        
        for fr in range(1, max_fr):  # NOTE: skips actual last frame
            # read in image
            fpath = os.path.join(args.data, "Laser", 
                                 "{:06d}.bin".format(fr))
            lsr = _to_torch(read_binimage(fpath))
            # normalize to [0,1] then do stats
            lsr = (lsr.float() - lsr.min().float()) / \
                (lsr.max().float() - lsr.min().float())
            if train_args.data_global_stats:
                lsr = normalize(lsr)
            else:
                lsr = (lsr - lsr.mean()) / lsr.std()
            # do full-frame prediction w/ overlap tile method
            lsr = lsr.to(device)
            if args.output_format == "watershed":
                ws = overlap_tile(lsr, net,
                                  crop_size=train_args.crop_size,
                                  pad_size=train_args.input_pad//2,
                                  output="watershed",
                                  **ws_kwargs)
                # move to numpy and force onto cpu
                ws = ws.detach().cpu().numpy()
                # save to *.mat format
                savemat(os.path.join(args.output, "frame{:06d}.mat".format(fr)),
                        {"W" : ws})
            else:
                cd, nd = overlap_tile(lsr, net,
                                      crop_size=train_args.crop_size,
                                      pad_size=train_args.input_pad//2,
                                      output="distance")
                # move to numpy, force onto cpu
                cd = cd.detach().cpu().numpy()
                nd = nd.detach().cpu().numpy()
                # save to *.mat format
                savemat(os.path.join(args.output, "frame{:06d}.mat".format(fR)),
                        {"cell_dist" : cd, "neig_dist" : nd})
            if not args.quiet:
                print("saved frame {:d}".format(fr))
    else:
        raise Exception("invalid data format {:s}".format(args.data_format))
    
        

def read_vk4image(fpath, im_type='light'):
    """read in an image from a VK4 file
    """
    assert im_type in ('light', 'height')
    with open(fpath, 'rb') as f:
        off = vk4e.extract_offsets(f)
        lsr_dict = vk4e.extract_img_data(off, im_type, f)
        lsr_data = lsr_dict['data']
        im_hgt = lsr_dict['height']
        im_wid = lsr_dict['width']
        lsr = np.reshape(lsr_data, (im_hgt, im_wid))
    return lsr


def read_binimage(fpath,im_w=1024,im_h=768):
    """read in an image from a binary file
    """
    lsr_data = np.fromfile(fpath,dtype=np.uint16)
    lsr = np.reshape(lsr_data,(im_h,im_w),'C')
    return lsr


def _to_torch(img):
    """convert uint16 image to [0,1] torch.FloatTensor
    """
    return torch.from_numpy(img.astype('float32')).unsqueeze(0) / 65535


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet timelapse movie processing")

    parser.add_argument("-c", "--checkpoint", type=str,
                        default="../../../expts/2021-08-30_wce", 
                        help="path to checkpoint folder w/ trained net")
    # input data args
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to root of folder with data to predict")
    parser.add_argument("-df", "--data-format", type=str, required=True,
                        choices=["ktl","kc"],
                        help="format of time-series dataset")
    # training data
    parser.add_argument("-td", "--training-data", type=str, default=None,
                        help="path to dataset used to train the net with")
    # output args
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="path to output folder")
    parser.add_argument("-of", "--output-format", type=str, default="distance",
                        choices=["watershed","distance"],
                        help="output predicted cell/neighbor distances or watershed")
    # watershed args
    parser.add_argument("-gs", "--gauss-sigma", type=float, default=1.0,
                        help="std. dev. of gaussian to blur net results with")
    parser.add_argument("-rm", "--rho-mask", type=float, default=0.2,
                        help="rho_seed parameter for watershed segmentation")
    parser.add_argument("-rs", "--rho-seed", type=float, default=0.2,
                        help="rho_seed parameter for watershed segmentation")
    parser.add_argument("-sp", "--seed-neig-power", type=float, default=2.0,
                        help="power to raise neighbor values when finding seeds")
    # misc. args
    parser.add_argument("-nc", "--no-cuda", action="store_true", default=False,
                        help="dont use cuda, even if available")
    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="dont show logging messages")
    ec = main(**vars(parser.parse_args()))
    exit(ec)
