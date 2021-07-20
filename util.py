"""Utility classes/functions
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def one_hot(labels, num_class, device, dtype, eps=1e-6):
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


def overlap_tile(img, net, crop_size, pad_size):
    """
    predict segmentation of `img` using the overlap-tile strategy

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
    pred : torch.Tensor
        the segmentation (

    """
    assert img.shape[-2] % crop_size == 0
    assert img.shape[-1] % crop_size == 0
    # figure out which device stuff is on
    dev = next(net.parameters()).device
    # pad input image
    img_pad = TF.pad(img, [pad_size, pad_size], padding_mode='reflect')
    tile_size = crop_size + pad_size  # size of tiles input to network
    
    pred = torch.zeros(img.shape[-2], img.shape[-1], dtype=torch.int64,
                       device=dev)
    for r in range(0, img.shape[-2], crop_size):
        for c in range(0, img.shape[-1], crop_size):
            tile = TF.crop(img_pad, r, c, tile_size, tile_size)
            tile_pred = F.softmax(net(tile), dim=1)
            pred[r:r+crop_size,c:c+crop_size] = torch.argmax(tile_pred, dim=1)
    return pred


def truefalse_posneg_stats(y_true, y_pred, num_class):
    # convert truth mask to one-hot
    if len(y_true.shape) == 2:
        y_true = y_true.unsqueeze(0)  # add fake "batch" dimension
    if len(y_true.shape) == 3:  # assume its (BxHxW) label mask
        oh_true = one_hot(y_true, num_class, y_pred.device, torch.float)
    else:
        assert len(y_true.shape) == 4 # assume its (BxCxHxW) one-hot encoding
        oh_true = y_true
    # make sure y_pred is also one-hot (same logic as above)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.unsqueeze(0)
    if len(y_pred.shape) == 3:
        oh_pred = one_hot(y_pred, num_class, y_pred.device, torch.float)
    else:
        assert len(y_pred.shape) == 4
        oh_pred = y_pred
        
    # compute statistics 
    # (each should be BxCxHxW and then summed to only leave channel)
    true_pos = (oh_pred * oh_true).sum(dim=(0,-2,-1))
    false_pos = (oh_pred * (1 - oh_true)).sum(dim=(0,-2,-1))
    true_neg = ((1 - oh_pred) * (1 - oh_true)).sum(dim=(0,-2,-1))
    false_neg = ((1 - oh_pred) * oh_true).sum(dim=(0,-2,-1))
    
    return (true_pos, false_pos), (true_neg, false_neg)