"""Utility classes/functions
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import numpy as np
from scipy import ndimage as ndi
import skimage.filters as skimf
from skimage.segmentation import watershed as skimage_watershed

def one_hot(labels, num_class, device, dtype, eps=1e-12):
    """
    Convert the input label vector/image to its equivalent one-hot encoding
    Parameters
    ----------
    labels : torch.LongTensor
        input label image/vector to convert to one-hot
    num_class : int
        number of classes
    device : torch.device
        DESCRIPTION.
    dtype : torch.dtype
        output datatype
    eps : float, optional
        epsilon value to prevent weirdness. The default is 1e-6.

    Raises
    ------
    ValueError
        invalid label shape.
    TypeError
        if input is not LongTensor.

    Returns
    -------
    torch.Tensor
        one-hot encoding of input label image/vector.

    """
    # input integrity
    if not len(labels.shape) == 3:
        raise ValueError(
            "invalid label shape, should be BxHxW. Got {}".format(
            labels.size()))
    if not labels.dtype == torch.int64:
        raise TypeError("labels must be type torch.int64 (Long)")
    batch_size, num_row, num_col = labels.size()
    one_hot = torch.zeros(batch_size, num_class, num_row, num_col,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def _ensure4d(img):
    """
    adjust input image so that it's 4-dimensional/okay for net
    """
    if len(img.shape) == 1:
        raise Exception("dont know how to handle 1d inputs")
    elif len(img.shape) == 2:
        return img.unsqueeze(0).unsqueeze(0)
    elif len(img.shape) == 3:
        return img.unsqueeze(0)
    elif len(img.shape) == 4:
        return img
    else:
        raise Exception("dont know how to handle >4d inputs")


def convTranspose2dOutputSize(dim, kernel_size, stride=1, padding=0, 
                              dilation=1, output_padding=0):
    return (dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) \
        + output_padding + 1
        

def conv2dOutputSize(dim, kernel_size, stride=1, padding=0, dilation=1):
    return ((dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) \
        + 1

def watershed(p_cell, p_neig,
              gauss_sigma=(1.5,1.5),
              rho_mask=0.09,
              rho_seed=0.5,
              seed_neig_power=2):
    """Post-processing watershed for net outputs

    default parameters are those given in paper
    """
    # if input is torch tensors, convert to numpy
    if torch.is_tensor(p_cell):
        p_cell = p_cell.detach().numpy()
    if torch.is_tensor(p_neig):
        p_neig = p_neig.detach().numpy()
    # apply gaussian blur to predictions
    phat_cell = skimf.gaussian(p_cell, gauss_sigma)
    phat_neig = skimf.gaussian(p_neig, gauss_sigma)
    # find region to flood & seeds for watershed
    p_mask = phat_cell > rho_mask
    p_seed = (p_cell - np.power(p_neig, seed_neig_power)) > rho_seed
    marks, _ = ndi.label(p_seed)
    return skimage_watershed(-p_cell, markers=marks, mask=p_mask)


class AvgValueTracker(object):
    """Computes/stores average & current value
    
    NOTE: ripped from PyTorch imagenet example
    """
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
        
    
class ProgressShower(object):
    """Show progress of training
    
    NOTE: ripped from PyTorch imagenet example
    """
    def __init__(self, n_batch, trackers, prefix=""):
        self.fmt = self._get_batch_fmtstr(n_batch)
        self.trackers = trackers
        self.prefix = prefix
    
    def display(self, batch):
        entries = [self.prefix + self.fmt.format(batch)]
        entries += [str(t) for t in self.trackers]
        print("\t".join(entries))
    
    def _get_batch_fmtstr(self, n_batch):
        n_dig = len(str(n_batch // 1))
        fmt = "{:" + str(n_dig) + "d}"
        return "[" + fmt + "/" + fmt.format(n_batch) + "]"
