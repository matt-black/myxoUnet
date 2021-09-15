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

        
class MaskRCNNDataset(torch.utils.data.Dataset):
    
    def __init__(self, base_dir, set_type, transform=None,
                 stat_global=True):
        # confirm input is ok
        assert set_type in ("train", "test")
        # setup directory & table locations
        self.img_dir = os.path.join(base_dir, set_type, "img")
        self.msk_dir = os.path.join(base_dir, set_type, "msk")
        self.tbl = pd.read_csv(os.path.join(base_dir, 
                                            "{}.csv".format(set_type)))
        # dataset length
        self.n_img = len(self.tbl.idx)
        # transforms
        self.to_tensor = transforms.PILToTensor()
        self.trans = transform

        self.stat_global = stat_global
        if stat_global:
            vals = np.array([])
            for i in range(self.n_img):
                im = np.array(self._get_image(i))
                vals = np.concatenate((vals, im.flatten()))
            vals = vals.astype(np.float32) / 65535
            self.normalize = transforms.Normalize(
                (np.mean(vals)), (np.std(vals)))
        
    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        # load image and corresponding label mask
        img = self.to_tensor(self._get_image(idx)) / 65535
        lbl = self.to_tensor(self._get_lblmask(idx))
        # apply transform
        if self.trans is not None:
            comb = torch.cat([img, lbl], dim=0)
            comb = self.trans(comb)
            img, lbl = split_imgmask(comb)
        # normalize image to mean/sd
        if self.stat_global:
            img = self.normalize(img)
        else:
            mu = img.float().mean()
            sd = img.float().std()
            img = transforms.functional.normalize(img, [mu], [sd])
        # map to range [0,1]
        img = (img - img.min()) / \
            (img.max()-img.min())

        # process label mask to generate target dictionary
        lbl = lbl.numpy()
        # get all the unique ids
        obj_ids = np.unique(lbl)
        obj_ids = obj_ids[1:]   # first id is background
        n_obj = len(obj_ids)
        # generate masks for each individual object
        pmsks = lbl == obj_ids[:, None, None]
        # figure out bounding boxes for each object
        boxes = []
        masks = []
        n_vobj = 0
        for i in range(n_obj):
            pos = np.where(pmsks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # TODO: figure out why i need to do this to filter out bad boxes
            if xmin != xmax and ymin != ymax:
                masks.append(pmsks[i])
                boxes.append([xmin, ymin, xmax, ymax])
                n_vobj += 1
        # convert everything to torch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((n_vobj,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img_id = torch.tensor([idx])
        try:
            area = (boxes[:,3] - boxes[:,1]) * \
                (boxes[:,2] - boxes[:,0])
        except:                 # if no objects in FOV
            print("[WARN] found no objects, so just going to try again")
            return self.__getitem__(idx) # just retry
        # flag to ignore certain entries
        # TODO: actually need this?
        iscrowd = torch.zeros((n_vobj,), dtype=torch.int64)

        # formulate output target dictionary
        target = {
            "boxes" : boxes,
            "labels" : labels,
            "masks" : masks,
            "image_id" : img_id,
            "area" :  area,
            "iscrowd" : iscrowd
        }
        return img, target
    
    def _get_image(self, idx):
        img_path =os.path.join(self.img_dir,
                               "im{:03d}.png".format(self.tbl.idx[idx]))
        return Image.open(img_path)

    def _get_lblmask(self, idx):
        msk_path =os.path.join(self.msk_dir,
                               "im{:03d}_clbl.png".format(self.tbl.idx[idx]))
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
