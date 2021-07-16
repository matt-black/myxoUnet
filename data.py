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
import pandas as pd


class MaskDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, base_dir, set_type="train", mask_type="j4"):
        self.img_dir = os.path.join(base_dir, set_type, "img")
        self.msk_dir = os.path.join(base_dir, set_type, "msk")
        self.tbl = pd.read_csv(os.path.join(base_dir, 
                                            "{}.csv".format(set_type)))
        if mask_type.lower() == "j4":
            self.msk_fmt = "im{:03d}_j4.png"
        elif mask_type.lower() == "j3":
            self.msk_fmt = "im{:03d}_j3.png"
        elif mask_type.lower() == "ce":
            self.msk_fmt = "im{:03d}_cells.png"
        else:
            raise Exception("invalid mask type")
        
    def __len__(self):
        return len(self.tbl.idx)

    def __getitem__(self, idx):
        msk_trans = transforms.Compose([transforms.PILToTensor(), 
                                        RgbLabelToMask()])
        img_trans = transforms.Compose([transforms.PILToTensor(),
                                        NormalizeInt32()])
        img = img_trans(self.get_image(idx))
        msk = msk_trans(self.get_mask(idx))
        return img, msk
    
    def get_image(self, idx):
        img_path = os.path.join(self.img_dir,
                                "im{:03d}.png".format(self.tbl.idx[idx]))
        return Image.open(img_path)
    
    def get_mask(self, idx):
        msk_name = self.msk_fmt.format(self.tbl.idx[idx])
        msk_path = os.path.join(self.msk_dir, msk_name)
        return Image.open(msk_path)
    

class NormalizeInt32(object):
    """Transform to normalize an input int32 image to [0,1]
    """
    def __call__(self, pic):
        maxval = torch.max(pic).float()
        minval = torch.min(pic).float()
        return (pic.float() - minval) / (maxval - minval)
    
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