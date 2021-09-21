"""Loss functions
"""

import torch
from torch import nn


class BUNetLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, l_r="mse"):
        super(BUNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # cross entropy loss for boundaries & fusion
        self.bce = nn.BCEWithLogitsLoss()
        self.lr_type = l_r
        # MSE loss for cell distances
        if l_r == "mse":
            self.lr = nn.MSELoss()
        elif l_r == "bce":
            self.lr = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("invalid type for l_r")

    def forward(self,
                fuse_pred, fuse_target,
                cd_pred, cd_target,
                bnd_pred, bnd_target):
        # fusion and boundary loss is cross entropy
        L_F = self.bce(fuse_pred, fuse_target.unsqueeze(0).float())
        L_B = self.bce(bnd_pred, bnd_target.unsqueeze(0).float())
        L_R = self.lr(cd_pred, cd_target)
        loss = self.gamma * L_F + self.beta * L_B + self.alpha * L_R
        return loss, (L_F, L_B, L_R)
