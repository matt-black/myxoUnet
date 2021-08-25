"""Loss functions
"""

import torch
from torch import nn
import torch.nn.functional as F

from util import one_hot


class DcanLoss(nn.Module):
    """Loss function from paper, just sum of data error for contour and mask
    """
    def __init__(self, eps=1e-12):
        super(DcanLoss, self).__init__()
        self.eps = eps
    
    def forward(self, p0, tar0, pc, tarc):
        msk_err = dcan_data_error(p0, tar0)
        cnt_err = dcan_data_error(pc, tarc)
        return msk_err + cnt_err
        

class DcanDataError(nn.Module):
    """Module wrapper around `dcan_data_error`
    """
    def __init__(self, eps):
        super(DcanDataError, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        return dcan_data_error(pred, target, self.eps)
        

def dcan_data_error(pred, target, eps=1e-12):
    """data error terms in loss function for DCAN
    take all pixels in the target labeled "1" and sum -log(prob) of predictions
    at those pixels
    """
    mask = target.ge(1-eps)
    return -torch.sum(torch.log(torch.masked_select(pred, mask)))


class DiceLoss(nn.Module):
    
    def __init__(self, reduction="mean", eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6
        self.reduction = reduction
        
    def forward(self, pred, target):
        return dice_loss(pred, target, self.reduction, self.eps)


def dice_loss(pred, target, reduction="mean", eps=1e-6):
    """
    Criterion using the Sorensen-Dice coefficient

    Parameters
    ----------
    pred : torch.Tensor
        (B, C, H, W) probability map of class predictions
    target : torch.Tensor
        (B, H, W) ground-truth class label tensor
        dtype should be torch.int64 (Long)
        
    Returns
    -------
    Dice loss between predictions and target
    """
    _check_valid_reduction(reduction)
    # convert target to one hot
    target_1h = one_hot(target, pred.size(1), 
                        device=pred.device, dtype=pred.dtype)
    # compute intersection/cardinality
    inter = torch.sum(pred * target_1h, (1, 2, 3))
    card  = torch.sum(pred + target_1h, (1, 2, 3))
    dice  = 2 * inter / (card + eps)
    # choose reduction
    if reduction == "mean":
        return torch.mean(1.0 - dice)
    elif reduction == "sum":
        return torch.sum(1.0 - dice)
    else:
        return (1.0 - dice)
    
class DiceRegularizedCrossEntropy(nn.Module):
    
    def __init__(self, reduction="mean", eps=1e-6):
        super(DiceRegularizedCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, pred, target):
        return dice_regularized_cross_entropy(pred, target,
                                              reduction=self.reduction,
                                              eps=self.eps)
    
    
def dice_regularized_cross_entropy(pred, target, reduction="mean", eps=1e-6):
    dice = dice_loss(pred, target, reduction="none", eps=eps)
    ce = F.cross_entropy(_collapse_outer_dims(pred), 
                         _collapse_outer_dims(target), 
                         reduction="none")
    ce = ce.mean(dim=-1)
    if reduction == "mean":
        return (dice + ce).mean()
    elif reduction == "sum":
        return (dice + ce).sum()
    else:
        return dice + ce


def _collapse_outer_dims(x):
    """
    collapse all dims past the first two into a single one

    Parameters
    ----------
    x : torch.Tensor
        (BxCx...)
    Returns
    -------
    `x` reshaped to (BxCxN)

    """
    assert len(x.shape) == 3 or len(x.shape) == 4
    if len(x.shape) == 4:
        new_shape = (x.shape[0], x.shape[1],
                     torch.mul(*x.shape[2:]))
    else:
         new_shape = (x.shape[0], torch.mul(*x.shape[1:]))   
    return torch.reshape(x, new_shape)


def _check_valid_reduction(reduction):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            "invalid reduction, {}. Valid are 'mean', 'sum', 'none'".format(
                reduction))    

