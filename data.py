"""Datasets
"""
# base python
import os
import math

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
                 transform=None, nplicates=1, stat_norm=True):
        # confirm input is ok
        assert set_type in ("train", "test")
        # setup directory & table locations
        self.img_dir = os.path.join(base_dir, set_type, "img")
        self.msk_dir = os.path.join(base_dir, set_type, "msk")
        self.tbl = pd.read_csv(os.path.join(base_dir, 
                                            "{}.csv".format(set_type)))
        # dataset length
        self.n_img = len(self.tbl.idx)
        self.n_class = n_classes
        self.nplicates = nplicates
        self.stat_norm = stat_norm  # should we normalize by mean/std?
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
        img = self.to_tensor(self._get_image(real_idx))
        if self.stat_norm:
            subval = img.float().mean()
            denom = img.float().std()
        else:
            subval = img.min().float()
            denom = img.max().float() - subval
            
        mask = self.to_tensor(self._get_mask(real_idx))
        # apply transformations
        if self.trans is not None:
            comb = torch.cat([img, mask], dim=0)  # concat along channel dim
            comb = self.trans(comb)
            img, mask = split_imgmask(comb)
        # normalize image and return
        img = (img.float() - subval) / denom
        return img, mask.long()
    
    def _get_image(self, idx):
        img_path = os.path.join(self.img_dir,
                                "im{:03d}.png".format(self.tbl.idx[idx]))
        return Image.open(img_path)
    
    def _get_mask(self, idx):
        msk_name = self.msk_fmt.format(self.tbl.idx[idx])
        msk_path = os.path.join(self.msk_dir, msk_name)
        return Image.open(msk_path)
    
    def class_percents(self):
        """Compute percentage of each class making up dataset
        """
        class_tot = [0 for i in range(self.n_class)]
        total_pix = 0
        for idx in range(self.n_img):
            msk = self.to_tensor(self._get_mask(idx))
            total_pix += torch.mul(msk.shape[-2], msk.shape[-1])
            for ci in range(self.n_class):
                class_tot[ci] += (msk == ci).sum()
        return torch.stack(class_tot) / total_pix * 100


class SizeScaledMaskDataset(torch.utils.data.Dataset):
    
    def __init__(self, base_dir, set_type, n_classes=2, crop_dim=256, 
                 transform=None, stat_norm=True):
        # confirm input is ok
        assert set_type in ("train", "test")
        # setup directory & table locations
        self.img_dir = os.path.join(base_dir, set_type, "img")
        self.msk_dir = os.path.join(base_dir, set_type, "msk")
        self.tbl = pd.read_csv(os.path.join(base_dir, 
                                            "{}.csv".format(set_type)))
        # dataset length
        self.n_img = len(self.tbl.idx)
        self.n_class = n_classes
        self.stat_norm = stat_norm  # should we normalize by mean/std?
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
        self.trans = transforms.Compose([transforms.RandomCrop(crop_dim), 
                                         transform])
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
        if self.stat_norm:
            subval = img.float().mean()
            denom = img.float().std()
        else:
            subval = img.min().float()
            denom = img.max().float() - subval
            
        mask = self.to_tensor(self._get_mask(tbl_idx))
        # apply transformations
        if self.trans is not None:
            comb = torch.cat([img, mask], dim=0)  # concat along channel dim
            comb = self.trans(comb)
            img, mask = split_imgmask(comb)
        # normalize image and return
        img = (img.float() - subval) / denom
        return img, mask.long()
    
    def _get_image(self, idx):
        img_path = os.path.join(self.img_dir,
                                "im{:03d}.png".format(self.tbl.idx[idx]))
        return Image.open(img_path)
    
    def _get_mask(self, idx):
        msk_name = self.msk_fmt.format(self.tbl.idx[idx])
        msk_path = os.path.join(self.msk_dir, msk_name)
        return Image.open(msk_path)
    
    def class_percents(self):
        """Compute percentage of each class making up dataset
        """
        class_tot = [0 for i in range(self.n_class)]
        total_pix = 0
        for idx in range(self.n_img):
            msk = self.to_tensor(self._get_mask(idx))
            total_pix += torch.mul(msk.shape[-2], msk.shape[-1])
            for ci in range(self.n_class):
                class_tot[ci] += (msk == ci).sum()
        return torch.stack(class_tot) / total_pix * 100


class DCANDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, set_type, 
                 transform=None, stat_norm=True):
        # confirm input is ok
        assert set_type in ("train", "test")
        # setup directory & table locations
        self.img_dir = os.path.join(base_dir, set_type, "img")
        self.msk_dir = os.path.join(base_dir, set_type, "msk")
        self.tbl = pd.read_csv(os.path.join(base_dir, 
                                            "{}.csv".format(set_type)))
        # dataset length
        self.n_img = len(self.tbl.idx)
        self.stat_norm = stat_norm  # should we normalize by mean/std?
        # setup formatting for input masks
        self.cellmsk_fmt = "im{:03d}_cell.png"
        self.cntrmsk_fmt = "im{:03d}_cntr.png"
        # image transforms
        self.to_tensor = transforms.PILToTensor()
        self.trans = transform
        
    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        img = self.to_tensor(self._get_image(idx))
        # do image normalization
        if self.stat_norm:
            subval = img.float().mean()
            denom = img.float().std()
        else:
            subval = img.min().float()
            denom = img.max().float() - subval
            
        # get edge, contour masks
        cellmask = self.to_tensor(self._get_cell_mask(idx))
        cntrmask = self.to_tensor(self._get_cntr_mask(idx))
        # apply transformations
        if self.trans is not None:
            comb = torch.cat([img, cellmask, cntrmask], dim=0)  # concat along channel dim
            comb = self.trans(comb)
            img, cellmask, cntrmask = split_img2mask(comb)
        # normalize image and return
        img = (img.float() - subval) / denom
        return img, cellmask.long(), cntrmask.long()
    
    def _get_image(self, idx):
        img_path = os.path.join(self.img_dir,
                                "im{:03d}.png".format(self.tbl.idx[idx]))
        return Image.open(img_path)
    
    def _get_cell_mask(self, idx):
        msk_name = self.cellmsk_fmt.format(self.tbl.idx[idx])
        msk_path = os.path.join(self.msk_dir, msk_name)
        return Image.open(msk_path)
    
    def _get_cntr_mask(self, idx):
        msk_name = self.cntrmsk_fmt.format(self.tbl.idx[idx])
        msk_path = os.path.join(self.msk_dir, msk_name)
        return Image.open(msk_path)
    
    
    
def _flatten_list(l):
    """flatten a list of lists into just a list
    """
    return [item for sublist in l for item in sublist]


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


def split_img2mask(img_2mask_tensor):
    """split concatenated image/mask from DCANDataset back out to
    an image and the two masks (cell and contour)
    """
    if len (img_2mask_tensor.size()) == 3:
        img = img_2mask_tensor[0,:,:].unsqueeze(0)
        cell_mask = img_2mask_tensor[1,:,:]
        cntr_mask = img_2mask_tensor[2,:,:]
    else:        
        img = img_2mask_tensor[:,0,:,:].unsqueeze(0)
        cell_mask = img_2mask_tensor[:,1,:,:]
        cntr_mask = img_2mask_tensor[:,2,:,:]
    return img, cell_mask, cntr_mask
    

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
