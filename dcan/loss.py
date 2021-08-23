"""Loss functions
"""

import torch
from torch import nn
import torch.nn.functional as F

from util import one_hot


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
