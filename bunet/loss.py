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
        self.ce = nn.CrossEntropyLoss()
        self.lr_type = l_r
        # MSE loss for cell distances
        if l_r == "mse":
            self.lr = nn.MSELoss()
        elif l_r == "bce":
            self.sig = nn.Sigmoid()
            self.lr = nn.BCELoss()
        else:
            raise ValueError("invalid type for l_r")

    def forward(self,
                fuse_pred, fuse_target,
                cd_pred, cd_target,
                bnd_pred, bnd_target):
        # fusion and boundary loss is cross entropy
        L_F = self.ce(fuse_pred, fuse_target)
        L_B = self.ce(bnd_pred, bnd_target)
        L_R = self.lr(self.sig(cd_pred), cd_target)
        loss = self.gamma * L_F + self.beta * L_B + self.alpha * L_R
        return loss, (L_F, L_B, L_R)
