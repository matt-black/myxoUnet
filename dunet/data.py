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
    def __init__(self, base_dir, set_type, transform=None, img_transform=None):
        # confirm input is ok
        assert set_type in ("train", "test")
        # setup directory & table locations
        self.img_dir = os.path.join(base_dir, set_type, "img")
        self.dst_dir = os.path.join(base_dir, set_type, "dst")
        self.tbl = pd.read_csv(os.path.join(base_dir, 
                                            "{}.csv".format(set_type)))
        # dataset length
        self.n_img = len(self.tbl.idx)
        # image transforms
        self.to_tensor = transforms.PILToTensor()
        self.trans = transform
        self.img_trans = img_transform
        
    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        real_idx = idx % self.n_img
        # fetch image & distances
        img = self.to_tensor(self._get_image(real_idx))
        cell_dist, neig_dist = self._get_dist(real_idx)
        # normalize image to range [-1, 1]
        img = (img.float() - img.float().min()) / \
            (img.float().max() - img.float().min())
        img = 2 * img - 1
        # apply transforms to image and mask, then split back out
        if self.trans is not None:
            comb = torch.cat ([img, cell_dist, neig_dist], dim=0)
            comb = self.trans(comb)
            img, cell_dist, neig_dist = split_img2dist(comb)
        cell_dist = cell_dist.unsqueeze(0)
        neig_dist = neig_dist.unsqueeze(0)
        # apply image-only transforms
        if self.img_trans is not None:
            img = self.img_trans(img)
        # return
        return img.float(), cell_dist.float(), neig_dist.float()
    
    def _get_image(self, idx):
        img_path = os.path.join(self.img_dir,
                                "im{:03d}.png".format(self.tbl.idx[idx]))
        return Image.open(img_path)

    def _get_dist(self, idx):
        dst_path = os.path.join(self.dst_dir,
                                "im{:03d}.mat".format(self.tbl.idx[idx]))
        dst = loadmat(dst_path)
        cell_dist = torch.from_numpy(dst["cell_dist"]).unsqueeze(0)
        neig_dist = torch.from_numpy(dst["neig_dist"]).unsqueeze(0)
        return cell_dist, neig_dist


class SizeScaledMaskDataset(torch.utils.data.Dataset):
    
    def __init__(self, base_dir, set_type, crop_dim=256, 
                 transform=None, img_transform=None):
        # confirm input is ok
        assert set_type in ("train", "test")
        # setup directory & table locations
        self.img_dir = os.path.join(base_dir, set_type, "img")
        self.msk_dir = os.path.join(base_dir, set_type, "msk")
        self.tbl = pd.read_csv(os.path.join(base_dir, 
                                            "{}.csv".format(set_type)))
        # dataset length
        self.n_img = len(self.tbl.idx)
        
        # image transforms
        self.to_tensor = transforms.PILToTensor()
        self.trans = transform
        self.img_trans = img_transform
        # figure out dataset sizing based on total # crops
        n_cropz = []
        for idx in range(self.n_img):
            w, h = Image.open(os.path.join(
                self.img_dir, "im{:03d}.png").format(
                    self.tbl.idx[idx])).size
            n_cropz.append(math.floor((w*h)/(crop_dim**2)))            
        id_list =[[i]*n for (i,n) in zip(range(self.n_img),n_cropz)]
        self.idx_list = _flatten_list(id_list)

    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        tbl_idx = self.idx_list[idx]
        img = self.to_tensor(self._get_image(tbl_idx))
        cell_dist, neig_dist = self._get_dist(tbl_idx)
        # normalize image to range [-1, 1]
        img = (img.float() - img.float().min()) / \
            (img.float().max() - img.float().min())
        img = 2 * img - 1
        # apply transforms to image and mask, then split back out
        if self.trans is not None:
            comb = torch.cat ([img, cell_dist, neig_dist], dim=0)
            comb = self.trans(comb)
            img, cell_dist, neig_dist = split_img2dist(comb)
        # apply image-only transforms
        if self.img_trans is not None:
            img = self.img_trans(img)
        # return
        return img, cell_dist.float(), neig_dist.float()
    
    def _get_image(self, idx):
        img_path = os.path.join(self.img_dir,
                                "im{:03d}.png".format(self.tbl.idx[idx]))
        return Image.open(img_path)
    
    def _get_dist(self, idx):
        dst_path = os.path.join(self.dst_dir,
                                "im{:03d}.mat".format(self.tbl.idx[idx]))
        dst = loadmat(dst_path)
        cell_dist = torch.from_numpy(dst["cell_dist"])
        neig_dist = torch.from_numpy(dst["neig_dist"])
        return cell_dist, neig_dist
    
    
def _flatten_list(l):
    """flatten a list of lists into just a list
    """
    return [item for sublist in l for item in sublist]


def split_img2dist(img_2dist_tensor):
    """split concatenated image/mask from DCANDataset back out to
    an image and the two masks (cell and contour)
    """
    if len (img_2dist_tensor.size()) == 3:
        img = img_2dist_tensor[0,:,:].unsqueeze(0)
        cell_dist = img_2dist_tensor[1,:,:]
        neig_dist = img_2dist_tensor[2,:,:]
    else:        
        img = img_2dist_tensor[:,0,:,:].unsqueeze(0)
        cell_dist = img_2dist_tensor[:,1,:,:]
        neig_dist = img_2dist_tensor[:,2,:,:]
    return img, cell_dist, neig_dist    


class NormalizeLong(object):
    """Transform to normalize an input Long image to [0,1]
    """
    def __call__(self, pic):
        maxval = torch.max(pic).float()
        minval = torch.min(pic).float()
        return (pic.float() - minval) / (maxval - minval)
    
    def __repr__(self):
        return self.__class__().__name__ + "()"
    

class AddGaussianNoise(object):

    def __init__(self, mean=0, std=1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + \
            self.mean

    def __repr__(self):
        return self.__class__.__name__ + \
            "(mean={0}, std={1})".format(self.mean, self.std)


class RandomContrastAdjust(object):
    def __init__(self, min_factor=0.1, max_factor=2):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, tensor):
        factor = random.uniform(self.min_factor,
                                self.max_factor)
        return transforms.functional.adjust_contrast(tensor, factor)

    def __repr__(self):
        return self.__class__.__name__ + \
            "(min={0}, max={1})".format(self.min_factor,
                                        self.max_factor)

    
class RgbLabelToMask(object):
    """Convert an RGB-label image to a mask
    
    Expects the input to be a torch tensor, so convention is [3xRxC]
    """
    def __call__(self, rgb):
        (m, _) = torch.max(rgb, 0)
        return (m > 0).long()
    
    def __repr__(self):
        return self.__class__().__name__ + "()"
