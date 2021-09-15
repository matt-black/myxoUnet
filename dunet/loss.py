import torch
from torch import nn
import matplotlib.pyplot as plt

class DUnetSmoothL1(nn.Module):
    """Loss function from paper
    Just take SmoothL1 from cell and neighbor distances, add them
    """

    def __init__(self, reduction="mean"):
        super(DUnetSmoothL1, self).__init__()
        self.smoothL1 = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, pred_cd, tar_cd, pred_nd, tar_nd):
        l1_cd = self.smoothL1(pred_cd, tar_cd)
        l1_nd = self.smoothL1(pred_nd, tar_nd)
        return l1_cd + l1_nd
