"""Datasets
"""
# base python
import os
import math
import random

# pytorch
import torch
from torchvision import transforms
# PIL
from PIL import Image

# other
import numpy as np
import pandas as pd
import elasticdeform.torch as elastic
from scipy.io import loadmat


class MaskDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, base_dir, set_type, transform=None, img_transform=None,
                 stat_global=True):
        # confirm input is ok
        assert set_type in ("train", "test")
        # setup directory & table locations
        self.img_dir = os.path.join(base_dir, set_type, "img")
        self.dst_dir = os.path.join(base_dir, set_type, "dst")
        self.msk_dir = os.path.join(base_dir, set_type, "msk")
        self.tbl = pd.read_csv(os.path.join(base_dir, 
                                            "{}.csv".format(set_type)))
        # dataset length
        self.n_img = len(self.tbl.idx)
        # image transforms
        self.to_tensor = transforms.PILToTensor()
        self.trans = transform
        self.img_trans = img_transform
        # global stats(?)
        self.stat_global = stat_global
        if self.stat_global:
            vals = np.array([])
            for i in range(self.n_img):
                im = self.to_tensor(self._get_image(i))
                im = norm2zero1(im).numpy()
                vals = np.concatenate((vals, im.flatten()))
            self.normalize = transforms.Normalize(
                (np.mean(vals)), (np.std(vals)))
            self.maxval = (1 - np.mean(vals)) / np.std(vals)
        else:
            self.maxval = 1.0

    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        real_idx = idx % self.n_img
        # fetch image & distances
        img = self.to_tensor(self._get_image(real_idx)).float()
        cell_mask = self.to_tensor(self._get_mask(real_idx))
        cell_mask = (cell_mask > 0).float()
        cell_dist = self._get_dist(real_idx).float()
        cell_bord = self.to_tensor(
            self._get_border_mask(real_idx)).float()
        
        
        # apply transforms to image and masks, then split back out
        if self.trans is not None:
            comb = torch.cat ([img, cell_mask, cell_dist, cell_bord], dim=0)
            comb = self.trans(comb)
            img, cell_mask, cell_dist, cell_bord = split_img3mask(comb)
        cell_dist = cell_dist.unsqueeze(0)

        # apply image-only transforms
        if self.img_trans is not None:
            img = self.img_trans(img)
        # normalize image, first to 0,1 then mean/std
        img = norm2zero1(img)
        if self.stat_global:
            img = self.normalize(img.float())
        else:
            img = (img.float() - img.float().mean()) / \
                img.float().std()
        # return
        return img, cell_mask.long(), cell_dist, cell_bord.long()
    
    def _get_image(self, idx):
        img_path = os.path.join(self.img_dir,
                                "im{:03d}.png".format(self.tbl.idx[idx]))
        return Image.open(img_path)

    def _get_mask(self, idx):
        msk_path = os.path.join(self.msk_dir,
                                "im{:03d}_clbl.png".format(self.tbl.idx[idx]))
        return Image.open(msk_path)
    
    def _get_border_mask(self, idx):
        msk_path = os.path.join(self.msk_dir,
                                "im{:03}_bord.png".format(self.tbl.idx[idx]))
        return Image.open(msk_path)

    def _get_dist(self, idx):
        dst_path = os.path.join(self.dst_dir,
                                "im{:03d}.mat".format(self.tbl.idx[idx]))
        dst = loadmat(dst_path)
        cell_dist = torch.from_numpy(dst["cell_dist"]).unsqueeze(0)
        return cell_dist


def norm2zero1(img_tens):
    """Normalize input image to range [0,1]
    """
    minval = img_tens.min().float()
    maxval = img_tens.max().float()
    return (img_tens.float() - minval) / (maxval - minval)


def split_img3mask(img_3mask_tensor):
    """split concatenated image/mask from DCANDataset back out to
    an image and the two masks (cell and contour)
    """
    if len (img_3mask_tensor.size()) == 3:
        img = img_3mask_tensor[0,:,:].unsqueeze(0)
        cell_mask = img_3mask_tensor[1,:,:]
        cell_dist = img_3mask_tensor[2,:,:]
        cell_bord = img_3mask_tensor[3,:,:]
    else:        
        img = img_3mask_tensor[:,0,:,:].unsqueeze(0)
        cell_mask = img_3mask_tensor[:,1,:,:]
        cell_dist = img_3mask_tensor[:,2,:,:]
        cell_bord = img_3mask_tensor[:,3,:,:]
    return img, cell_mask, cell_dist, cell_bord


class RandomRotateDeformCrop(object):
    """Applies a random rotation, crop and deformation to input
    """
    def __init__(self, sigma=10, points=3, crop=256):
        self.sigma = sigma
        self.shape = (2, points, points)
        self.crop_size = crop

    def __call__(self, img):
        # figure out crop parameters
        crop_par = transforms.RandomCrop.get_params(
            img, (self.crop_size, self.crop_size))
        crop_slcs = [slice(crop_par[0], crop_par[0]+crop_par[2]),
                     slice(crop_par[1], crop_par[1]+crop_par[3])]
        # random rotation
        ang = torch.rand(1) * 90
        # random deformation
        disp = torch.randn(*self.shape) * self.sigma
        n_ax_in = len (img.size())
        axes = (n_ax_in-2, n_ax_in-1) # only apply to outermost (image) axes
        return elastic.deform_grid(img, disp, order=0, mode="mirror",
                                   crop=crop_slcs, rotate=ang, axis=axes)
        
    def __repr__(self):
        return self.__class__().__name__ + "()"
