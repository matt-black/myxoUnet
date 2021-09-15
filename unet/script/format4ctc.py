#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format a segmented timelapse movie and its accompanying images into the format
required for the Cell Tracking Challenge
@author: matt
"""

# %% imports
import os, sys
import json
import argparse

# science libs
import numpy as np
from scipy.io import loadmat


def main(**kwargs):
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTC formatting script")
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to images directory")
    parser.add_argument("-df", "--data-format", type=str, required=True,
                        choices=["ktl","kc"],
                        help="format of time-series dataset")
    parser.add_argument("-s", "--segmentation-path", type=str, required=True,
                        help="path to segmentations at each frame")
    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="output directory (will be made if doesnt exist already)")
    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="dont show logging messages")
    ec = main(**vars(parser.parse_args()))
    exit(ec)
    
              
