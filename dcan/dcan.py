"""
"""

import torch
from torch import nn
import torch.nn.functional as F

class DCAN(nn.Module):
    """
    Implementation of
    DCAN: Deep Contour-Aware Networks for Accurate Gland Segmentation
    (Chen et al., 2016)
    https://arxiv.org/abs/1605.02677
    """
    def __init__(self,
                 in_channels,
                 depth=5,
                 wf=6,
                 kernel_size=3,
                 batch_norm=True,
                 up_mode='upconv',
                 **kwargs):
        super(DCAN, self).__init__()
        assert depth >= 3
        assert up_mode in ('upconv', 'upsample')
        # construct down-sampling path
        self.down_path = nn.ModuleList()
        # at last 3 steps, build up contour/mask upsampling paths
        self.up_mask = nn.ModuleList()
        self.up_cntr = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth):
            # determine # of output channels
            # all but 2nd-to-last layer consist of doubling
            if i == depth-2:
                out_channels = prev_channels
            else:
                out_channels = 2**(wf+i)
                
            ksze = kernel_size if i > depth-1 else 1
            self.down_path.append(
                DCANDownsampleBlock(prev_channels, out_channels, ksze,
                                    0, batch_norm)
            )
            # last 3 layers have associated deconvolution channels
            if up_mode == 'upsample':
                out_dim = kwargs['out_dim']
                if i >= depth-3:
                    self.up_mask.append(
                        DCANUpsampleInterpBlock(out_channels, out_dim)
                    )
                    self.up_cntr.append(
                        DCANUpsampleInterpBlock(out_channels, out_dim)
                    )
            else:
                out_dim = kwargs['out_dim']
                scale_factor = depth + 3
                if i < depth - 1:
                    up_kern = scale_factor * 2**(i-2)
                else:
                    up_kern = scale_factor * 2**(i-3)
                
                if i >= depth-3:
                    self.up_mask.append(
                        DCANUpsampleConvBlock(out_channels, up_kern, up_kern, 
                                              out_dim)
                    )
                    self.up_cntr.append(
                        DCANUpsampleConvBlock(out_channels, up_kern, up_kern, 
                                              out_dim)
                    )
            prev_channels = out_channels

    def forward(self, x):
        # do the downsampling
        ys = []
        for i, down in enumerate(self.down_path):
            x = down(x)
        
            # do max-pooling after convolution
            if i < (len(self.down_path) - 1):  # unless its the last layer
                x = F.max_pool2d(x, 2)
            # save the last 3 layer-outputs for deconvolution
            if i >= (len(self.down_path) - 3):
                ys.append(x)

        # parallel upsampling paths
        mask_outs = []
        for y, up in zip(ys, self.up_mask):
            mask_outs.append(up(y))
        mask = torch.sum(torch.stack(mask_outs, dim=0), dim=0)
        
        cntr_outs = []
        for y, up in zip(ys, self.up_cntr):
            cntr_outs.append(up(y))
        cntr = torch.sum(torch.stack(cntr_outs, dim=0), dim=0)
        
        # intermediate classifier
        interms = zip (mask_outs, cntr_outs)
        
        return mask, cntr, interms
        
        
class DCANDownsampleBlock(nn.Module):
    """
    Downsampling block consisting of a convolutional layer, activation,
    and (possible) batch normalization
    """
    def __init__(self,
                 in_size,
                 out_size,
                 kern_size=3,
                 padding=0,
                 batch_norm=True):
        super(DCANDownsampleBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=kern_size,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class DCANUpsampleConvBlock(nn.Module):
    """Upsampling block of DCAN network
    """
    def __init__(self, 
                 in_channels,
                 kernel_size,
                 stride, out_dim):
        super(DCANUpsampleConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, 2, 
                                     kernel_size=kernel_size,
                                     stride=stride)
        self.activ = nn.ReLU()
        self.conv = nn.Conv2d(2, 2, kernel_size=1)
        self.out_dim = out_dim
    
    def forward(self, x):
        out_sze = [x.shape[0], x.shape[1], self.out_dim, self.out_dim]
        up = self.activ(self.up(x, output_size=out_sze))
        return self.activ(self.conv(up))


class DCANUpsampleInterpBlock(nn.Module):
    """Upsampling block of DCAN network
    Consisting of a bilinear upsample then a convolution
    """
    def __init__(self, 
                 in_channels,
                 output_dim):
        super(DCANUpsampleInterpBlock, self).__init__()
        self.output_dim = output_dim
        self.activ = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)
        
    def forward(self, x):
        up = F.interpolate(x, size=self.output_dim, 
                           mode='bilinear',
                           align_corners=True)
        out = self.activ(self.conv(up))
        return out
