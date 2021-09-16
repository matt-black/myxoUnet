"""Utility classes/functions
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import numpy as np
from scipy import ndimage as ndi
from scipy import interpolate
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


def overlap_tile(img, net, crop_size, pad_size,
                 output="watershed", interp_edges=True,
                 **kwargs):
    """
    predict segmentation of `img` using the overlap-tile strategy
    NOTE: currently only works if the image is mod-divisible by `crop_size` in both dimensions
    
    Parameters
    ----------
    img : torch.Tensor
        the input image to predict segmentation of (CxHxW)
    net : torch.nn.Module
        the unet
    crop_size : int
        dimension of (square) region to be predicted in img
    pad_size : int
        amount to pad `crop_size` by to generate input tiles for network
    Returns
    -------
    pred/prob : torch.Tensor
        either class predictions (pred) or class probabilities (prob)
        if you specify "pred" the output is a (HxW) torch.LongTensor with pixel
        values as class ids
        if you specify "prob" the output is a (NxHxW) torch.FloatTensor with
        pixel values as class probabilities and each dimension corresponding
        to a single class

    """
    assert output in ("watershed", "distance")
    
    # unsqueeze input image to make it work with the net
    # (net wants BxCxHxW)
    img = _ensure4d(img)
    output_shape = [img.shape[-2], img.shape[-1]]

    # if crop doesn't cleanly fit into image size, pad image so it does
    if ((img.shape[-1] % crop_size) > 0) or ((img.shape[-2] % crop_size) > 0):
        if (img.shape[-1] % crop_size) > 0: # need to pad in row dimension
            col_pad = (crop_size - (img.shape[-1] % crop_size)) // 2
        else:
            col_pad = 0
        if (img.shape[-2] % crop_size) > 0: # need to pad in col dimension
            row_pad = (crop_size - (img.shape[-2] % crop_size)) // 2
        else:
            row_pad = 0
    else:
        row_pad = 0
        col_pad = 0

    # figure out which device stuff is on
    dev = next(net.parameters()).device
    # pad input image
    img_pad = TF.pad(img, [pad_size, pad_size], padding_mode='reflect')
    tile_size = crop_size + 2*pad_size  # size of tiles input to network
    cd_pred = torch.ones(3, img.shape[-2], img.shape[-1],
                         dtype=torch.float32, device=dev) * (-1)
    nd_pred = torch.ones(3, img.shape[-2], img.shape[-1],
                         dtype=torch.float32, device=dev) * (-1)
    net.eval()
    for r in range(0, img.shape[-2], crop_size):
        for c in range(0, img.shape[-1], crop_size):
            # adjust for padding size, then crop out tile
            tile = TF.crop(img_pad, r, c, tile_size, tile_size)
            # run tile through net, add it to crop
            cd, nd = net(tile)
            cd_pred[0,r+1:r+crop_size-1,c+1:c+crop_size-1] = cd[0,0,1:-1,1:-1]
            nd_pred[0,r+1:r+crop_size-1,c+1:c+crop_size-1] = nd[0,0,1:-1,1:-1]
    for r in range(crop_size//2, img.shape[-2]-crop_size//2, crop_size):
        for c in range(0, img.shape[-1], crop_size):
            # adjust for padding size, then crop out tile
            tile = TF.crop(img_pad, r, c, tile_size, tile_size)
            # run tile through net, add it to crop
            cd, nd = net(tile)
            cd_pred[1,r+1:r+crop_size-1,c+1:c+crop_size-1] = cd[0,0,1:-1,1:-1]
            nd_pred[1,r+1:r+crop_size-1,c+1:c+crop_size-1] = nd[0,0,1:-1,1:-1]
    for r in range(0, img.shape[-2], crop_size):
        for c in range(crop_size//2, img.shape[-1]-crop_size//2, crop_size):
            # adjust for padding size, then crop out tile
            tile = TF.crop(img_pad, r, c, tile_size, tile_size)
            # run tile through net, add it to crop
            cd, nd = net(tile)
            cd_pred[2,r+1:r+crop_size-1,c+1:c+crop_size-1] = cd[0,0,1:-1,1:-1]
            nd_pred[2,r+1:r+crop_size-1,c+1:c+crop_size-1] = nd[0,0,1:-1,1:-1]
    cd_pred, _ = torch.max(cd_pred, dim=0)
    nd_pred, _ = torch.max(nd_pred, dim=0)
    if interp_edges:
        # formulate mask to say which values need to be interpolated
        mask = torch.zeros(img.shape[-2], img.shape[-1],
                           dtype=torch.bool, device=dev)
        for r in range(crop_size, img.shape[-2], crop_size):
            mask[r,:] = True
        for c in range(crop_size, img.shape[-1], crop_size):
            mask[:,c] = True
        mask = mask.cpu().numpy()
        print(mask.shape)
        cd_pred = cd_pred.detach().cpu().numpy()
        nd_pred = nd_pred.detach().cpu().numpy()
        xx, yy = np.meshgrid(np.arange(img.shape[-1]),
                             np.arange(img.shape[-2]))
        # interpolate the cell distance data
        interp_cdz = interpolate.griddata((xx[~mask], yy[~mask]),
                                          cd_pred[~mask],
                                          (xx[mask], yy[mask]),
                                          method='linear',
                                          fill_value=0)
        cd_pred[yy[mask],xx[mask]] = interp_cdz
        # repeat for neigbor distances
        interp_ndz = interpolate.griddata((xx[~mask], yy[~mask]),
                                          nd_pred[~mask],
                                          (xx[mask], yy[mask]),
                                          method='linear',
                                          fill_value=0)
        nd_pred[yy[mask],xx[mask]] = interp_ndz
        # convert back to pytorch to match case where we dont interp
        cd_pred = torch.from_numpy(cd_pred)
        nd_pred = torch.from_numpy(nd_pred)
    if output == "watershed":
        rho_m = kwargs["rho_mask"]
        rho_s = kwargs["rho_seed"]
        seedp = kwargs["seed_neig_power"]
        sigma = kwargs["gauss_sigma"]
        return watershed(cd_pred, nd_pred,
                         gauss_sigma=sigma,
                         rho_mask=rho_m,
                         rho_seed=rho_s,
                         seed_neig_power=seedp)
    else:
        return cd_pred, nd_pred


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
    return skimage_watershed(-p_cell, markers=marks, mask=p_mask,
                             watershed_line=True)


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
