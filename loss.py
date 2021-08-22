"""Loss functions
"""

import torch
from torch import nn
import torch.nn.functional as F

from util import one_hot


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
    

class JLoss(nn.Module):
    
    def __init__(self, weight=None):
        super(JLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred, target):
        return j_loss(pred, target, self.weight)


def j_loss(pred, target, weights=None):
    """
    J-statistic based loss function

    from, arXiv:1910.09783v1

    Parameters
    ----------
    pred : torch.Tensor
        (B, C, H, W) probability map of class predictions
    target : torch.Tensor
        (B, H, W) ground-truth class label tensor
        dtype should be torch.int64 (Long)
        

    Returns
    -------
    J loss for prediction & targets
    """
    if weights is None:
        weights = torch.ones((pred.size(1), pred.size(1)), 
                             device=pred.device, dtype=pred.dtype)
    
    # convert target to one hot
    num_class = pred.size(1)
    target_1h = one_hot(target, num_class, 
                        device=pred.device, dtype=pred.dtype)
    
    # compute phi
    n_i = target_1h.sum(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1)
    phi = torch.div(target_1h, n_i)
    
    # compute per-class loss
    loss = torch.zeros((pred.size(0), num_class, num_class),
                       device=pred.device, dtype=pred.dtype)
    for ci in range(pred.size(1)):
        for ck in range(pred.size(1)):
            if ci == ck:
                continue
            delta_ik = torch.div(phi[:,ci,:,:]-phi[:,ck,:,:], 2)
            loss[:,ci,ck] = weights[ci,ck] * torch.log(
                0.5 + torch.mul(pred[:,ci,:,:], delta_ik).sum(dim=(-2,-1)))
    # sum over classes
    j = torch.neg(loss.sum(dim=(-2, -1)))
    return j
    
    
class JRegularizedCrossEntropyLoss(nn.Module):
    
    def __init__(self, j_wgt=None, ce_wgt=None):
        super(JRegularizedCrossEntropyLoss, self).__init__()
        self.j_wgt = j_wgt
        self.ce_wgt = ce_wgt

    def forward(self, pred, target):
        return j_regularized_cross_entropy(pred, target, 
                                           self.j_wgt, self.ce_wgt)
    
    
def j_regularized_cross_entropy(pred, target, j_wgt=None, ce_wgt=None):
    """
    """
    jl = j_loss(pred, target, j_wgt)
    if ce_wgt is None:
        ce = F.cross_entropy(_collapse_outer_dims(pred), 
                             _collapse_outer_dims(target), 
                             reduction="mean")
    else:
        ce = F.cross_entropy(_collapse_outer_dims(pred),
                             _collapse_outer_dims(target),
                             weight=ce_wgt, reduction="mean")    
    return jl + ce


class DcanLoss(nn.Module):
    def __init__(self, eps):
        super(DcanLoss, self).__init__()
        self.eps = eps
    
    def forward(self, p0, tar0, pc, tarc):
        msk_err = dcan_data_error(p0, tar0)
        cnt_err = dcan_data_error(pc, tarc)
        return msk_err + cnt_err
        

class DcanDataError(nn.Module):
    def __init__(self, eps):
        super(DcanDataError, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        return dcan_data_error(pred, target, self.eps)
        

def dcan_data_error(pred, target, eps=1e-6):
    """data error terms in loss function for DCAN
    take all pixels in the target labeled "1" and sum -log(prob) of predictions
    at those pixels
    """
    mask = target.ge(1-eps)
    return -torch.sum(torch.log(torch.masked_select(pred, mask)))
    

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
