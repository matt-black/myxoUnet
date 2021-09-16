"""dunet
"Distance UNET" from 10.1371/journal.pone.0243219

short story: its a normal unet with a two decoder branches that predicts cell and neighbor distances, respectively
"""
import torch
from torch import nn
import torch.nn.functional as F


class DUnet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            depth=5,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
            down_mode='conv'
    ):
        """
        Implementation of KIT-Sch-GE / 2021 Segmentation Challenge Dual U-Net
        journals.plos.org/plosone/article?id=10.1371/journal.pone.0243219

        Using the default arguments will yield the exact version used
        in the original paper

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
        super(DUnet, self).__init__()
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
        self.up_path_nd = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_cd.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i),
                            up_mode, padding, batch_norm)
            )
            self.up_path_nd.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i),
                            up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
        self.last_cd = nn.Conv2d(prev_channels, 1, kernel_size=1)
        self.last_nd = nn.Conv2d(prev_channels, 1, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, (down, samp) in enumerate(zip(self.down_path, self.down_conv)):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = samp(x)     # downsampling step
        y = x
        for i, (up_cd, up_nd) in enumerate(zip(self.up_path_cd, self.up_path_nd)):
            x = up_cd(x, blocks[-i - 1])
            y = up_nd(y, blocks[-i - 1])
        return self.last_cd(x), self.last_nd(y)

    
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
