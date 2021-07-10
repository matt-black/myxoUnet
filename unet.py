# 

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as fun

class DoubleConv2d(nn.Module):
    """does two sequential 2d convolutions
    """
    def __init__(self, chan_in, chan_out):
        super(DoubleConv2d, self).__init__()
        self.seq = nn.Sequential(
            # first convolution
            nn.Conv2d(chan_in, chan_out, 3, padding=1),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(inplace=True),
            # second convolution
            nn.Conv2d(chan_out, chan_out, 3, padding=1),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)


class DownConv2d(nn.Module):
    def __init__(self, chan_in, chan_out):
        super(DownConv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(chan_in, chan_out))
        
    def forward (self, x):
        return self.seq(x)


class UpConv2d(nn.Module):
    def __init__(self, chan_in, chan_out, bilinear=True):
        super(UpConv2d, self).__init__()
        hlf_in = chan_in // 2
        if bilinear:
            self.upsamp = nn.Upsample(scale_factor=2, mode="bilinear",
                                      align_corners=True)
        else:
            hlf_in = chan_in // 2
            self.upsamp = nn.ConvTranspose2d(hlf_in, hlf_in, 2,
                                             stride=2)
        self.conv = DoubleConv2d(chan_in, chan_out)

    def forward(self, x1, x2):
        x1 = self.upsamp(x1)
        # compute amt to pad image
        dY = x2.size()[2] - x1.size()[2]
        dX = x2.size()[3] - x1.size()[3]
        # do padding
        x1 = fun.pad(x1, (dX // 2, dX - dX // 2,
                          dY // 2, dY - dY // 2))
        # concat output 
        x = torch.cat ([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_chan, n_class):
        # basic setup
        super (UNet, self).__init__()
        self.n_chan = n_chan
        self.n_class = n_class
        # the input/downsampling part of the U
        self.inp = DoubleConv2d(n_chan, 64)
        self.down1 = DownConv2d(64, 128)
        self.down2 = DownConv2d(128, 256)
        self.down3 = DownConv2d(256, 512)
        self.down4 = DownConv2d(512, 512)
        # the upsampling/output part of the U
        self.up1 = UpConv2d(1024, 256)
        self.up2 = UpConv2d(512, 128)
        self.up3 = UpConv2d(256, 64)
        self.up4 = UpConv2d(128, 64)
        self.out = DoubleConv2d(64, n_class)

    def forward(self, x):
        # the input/downsampling part
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # the output/upsampling part
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
    
