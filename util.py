#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        epsilon value to prevent division by zero error. The default is 1e-6.

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
    TODO

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    net : TYPE
        DESCRIPTION.
    crop_size : TYPE
        DESCRIPTION.
    pad_size : TYPE
        DESCRIPTION.

    Returns
    -------
    proc : TYPE
        DESCRIPTION.

    """
    assert img.shape[-2] % crop_size == 0
    assert img.shape[-1] % crop_size == 0
    # figure out which device stuff is on
    dev = next(net.parameters()).device
    # pad input image
    img_pad = TF.pad(img, [pad_size, pad_size], padding_mode='reflect')
    tile_size = crop_size + pad_size  # size of tiles input to network
    
    proc = torch.zeros(img.shape[-2], img.shape[-1], dtype=torch.int64,
                       device=dev)
    for r in range(0, img.shape[-2], crop_size):
        for c in range(0, img.shape[-1], crop_size):
            tile = TF.crop(img_pad, r, c, tile_size, tile_size)
            pred = F.softmax(net(tile), dim=1)
            proc[r:r+crop_size,c:c+crop_size] = torch.argmax(pred, dim=1)
    return proc