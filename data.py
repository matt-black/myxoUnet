"""Datasets
"""
# base python
import os

# pytorch
import torch
from torchvision import transforms
# PIL
from PIL import Image

# other
import numpy as np
import pandas as pd
import elasticdeform.torch as elastic

class MaskDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, base_dir, set_type, n_classes=2, 
                 transform=None, nplicates=1):
        # confirm input is ok
        assert set_type in ("train", "test")
        # setup directory & table locations
        self.img_dir = os.path.join(base_dir, set_type, "img")
        self.msk_dir = os.path.join(base_dir, set_type, "msk")
        self.tbl = pd.read_csv(os.path.join(base_dir, 
                                            "{}.csv".format(set_type)))
        # dataset length
        self.n_img = len(self.tbl.idx)
        self.nplicates = nplicates
        # setup formatting for input masks
        if n_classes == 4:
            self.msk_fmt = "im{:03d}_j4.png"
        elif n_classes == 3:
            self.msk_fmt = "im{:03d}_j3.png"
        elif n_classes == 2:
            self.msk_fmt = "im{:03d}_cell.png"
        else:
            raise Exception("invalid mask number")
        # image transforms
        self.to_tensor = transforms.PILToTensor()
        self.trans = transform
        
    def __len__(self):
        return self.n_img * self.nplicates

    def __getitem__(self, idx):
        real_idx = idx % self.n_img
        img = self.to_tensor(self.get_image(real_idx))
        maxval = img.max().float()
        minval = img.min().float()
        mask = self.to_tensor(self.get_mask(real_idx))
        # apply transformations
        if self.trans is not None:
            comb = torch.cat([img, mask], dim=0)  # concat along channel dim
            comb = self.trans(comb)
            img, mask = split_imgmask(comb)
        # normalize to [0, 1] in range of entire image
        img = (img.float() - minval) / (maxval - minval)
        return img, mask.long()
    
    def get_image(self, idx):
        img_path = os.path.join(self.img_dir,
                                "im{:03d}.png".format(self.tbl.idx[idx]))
        return Image.open(img_path)
    
    def get_mask(self, idx):
        msk_name = self.msk_fmt.format(self.tbl.idx[idx])
        msk_path = os.path.join(self.msk_dir, msk_name)
        return Image.open(msk_path)


def split_imgmask(img_mask_tensor):
    """split the concatenated image/mask from MaskDataset back out to
    an image and a mask
    """
    if len (img_mask_tensor.size()) == 3:
        img = img_mask_tensor[0,:,:].unsqueeze(0)
        mask = img_mask_tensor[1,:,:]
    else:        
        img = img_mask_tensor[:,0,:,:].unsqueeze(0)
        mask = img_mask_tensor[:,1,:,:]
    return img, mask


class NormalizeLong(object):
    """Transform to normalize an input Long image to [0,1]
    """
    def __call__(self, pic):
        maxval = torch.max(pic).float()
        minval = torch.min(pic).float()
        return (pic.float() - minval) / (maxval - minval)
    
    def __repr__(self):
        return self.__class__().__name__ + "()"
    

class RandomElasticDeformation(object):
    """light wrapper around elasticdeform
    """
    def __init__(self, sigma=10, points=3):
        self.sigma = sigma
        self.shape = (2, points, points)
        
    def __call__(self, img):
        disp = torch.tensor(np.random.randn(*self.shape) * self.sigma)
        return elastic.deform_grid(img, disp, order=3, mode='mirror')
    
    def __repr__(self):
        return self.__class__().__name__ + "()"


class RandomRotateDeformCrop(object):
    """Applies a random rotation, crop and deformation to input
    """
    def __init__(self, sigma=10, points=3, crop=256):
        self.sigma = sigma
        self.shape = (2, points, points)
        self.crop_size = crop

    def __call__(self, img):
        # figure out crop parameters
        crp_tup = (self.crop_size, self.crop_size)
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


class RgbLabelToMask(object):
    """Convert an RGB-label image to a mask
    
    Expects the input to be a torch tensor, so convention is [3xRxC]
    """
    def __call__(self, rgb):
        (m, _) = torch.max(rgb, 0)
        return (m > 0).long()
    
    def __repr__(self):
        return self.__class__().__name__ + "()"
