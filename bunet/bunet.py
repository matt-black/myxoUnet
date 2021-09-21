"""bunet
"Border UNET" from "Dual U-Net for the Segmentation of Overlapping Glioma"

SEE: https://ieeexplore.ieee.org/abstract/document/8744511
"""
import torch
from torch import nn
import torch.nn.functional as F


class BUNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            depth=5,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
            down_mode='maxpool',
            cell_channels=1
    ):
        """
        Implementation of "Dual U-Net for the Segmentation of Overlapping Glioma"
        https://ieeexplore.ieee.org/abstract/document/8744511

        Args:
            in_channels (int): number of input channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(BUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        assert down_mode in ('conv', 'maxpool')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        # the downsampling path
        self.down_path = nn.ModuleList()
        self.down_conv = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
            # downsampling mode is either a conv or space-equivalent maxpool
            if down_mode == 'conv':
                self.down_conv.append(
                    nn.Conv2d(prev_channels, prev_channels,
                              kernel_size=3, stride=2)
                )
            else:
                self.down_conv.append(nn.MaxPool2d(2))
        
        # the upsampling path for cell distances
        self.up_path_cd = nn.ModuleList()
        self.up_path_bd = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.up_path_cd.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i),
                            up_mode, padding, batch_norm)
            )
            self.up_path_bd.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i),
                            up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
        self.last_cd = nn.Conv2d(prev_channels, cell_channels, kernel_size=1)
        self.last_bd = nn.Conv2d(prev_channels, 2, kernel_size=1)
        # the fusion block
        self.fuse = BUNetFusionBlock(cell_channels+2)

    def forward(self, x):
        blocks = []
        # do downsampling on common path
        for i, (down, samp) in enumerate(zip(self.down_path, self.down_conv)):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = samp(x)     # downsampling
        # now upsample along each parallel path
        y = x                   # x is cells, y is border
        for i, (up_cd, up_bd) in enumerate(zip(self.up_path_cd, self.up_path_bd)):
            x = up_cd(x, blocks[-i - 1])
            y = up_bd(y, blocks[-i - 1])
        x = self.last_cd(x)
        y = self.last_bd(y)
        # fusion
        fused = self.fuse(x, y)
        return fused, x, y


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class BUNetFusionBlock(nn.Module):
    
    def __init__(self, in_channels):
        super(BUNetFusionBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=1)
        )

    def forward(self, x_c, x_b):
        # 
        cat = torch.cat([x_c, x_b], 1)
        return self.convs(cat)
