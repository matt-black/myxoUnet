"""Loss functions
"""

import torch
from torch import nn


class BUNetLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(BUNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # cross entropy loss for boundaries & fusion
        self.ce = nn.CrossEntropyLoss()
        # MSE loss for cell distances
        self.mse = nn.MSELoss()

    def forward(self,
                fuse_pred, fuse_target,
                cd_pred, cd_target,
                bnd_pred, bnd_target):
        # fusion and boundary loss is cross entropy
        L_F = self.ce(fuse_pred, fuse_target)
        L_B = self.ce(bnd_pred, bnd_target)
        L_R = self.mse(cd_pred, cd_target)
        loss = self.gamma * L_F + self.beta * L_B + self.alpha * L_R
        return loss, (L_F, L_B, L_R)


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
