"""Training script
"""

# python
import os
import csv
import json
from math import inf
import time
import random
import argparse

# pytorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import center_crop

from unet import UNet
from data import MaskDataset, RandomRotateDeformCrop

def main(**kwargs):
    args = argparse.Namespace(**kwargs)

    # random seed?
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    # setup output folder
    if args.save_path is not None:
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        # save args to file
        with open(os.path.join(args.save_path, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    # use cuda?
    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # make loss function
    if args.loss == "j3":
        n_class = 3
        raise NotImplementedError()
    elif args.loss == "j4":
        n_class = 4
        raise NotImplementedError()
    else:
        n_class = 2
        crit = nn.CrossEntropyLoss()
    
    # setup UNet
    net = UNet(in_channels=1,
               n_classes=n_class,
               depth=args.unet_depth,
               wf=4,
               padding=args.unet_pad,
               batch_norm=args.unet_batchnorm,
               up_mode=args.unet_upmode)
    
    if not args.unet_pad:
        # determine correct crop size, if applicable
        test_inp = torch.randn(1, 1, args.crop_size, args.crop_size)
        out_size = 0
        pad = 0
        while out_size < args.crop_size:
            out_size = net(torch.randn(
                1, 1, args.crop_size+pad, args.crop_size+pad)).size(-1)
            pad = pad + 1
        crop_dim = args.crop_size+pad
        if crop_dim % 2 > 0:
            crop_dim = crop_dim + 1
    else:
        crop_dim = args.crop_size

    # setup optimizer
    opt = optim.SGD(net.parameters(), args.learning_rate)
    
    # build up train/test datasets
    if not (os.path.isdir(args.data)):
        raise Exception("specified data directory doesn't exist")
    datakw = {"num_workers" : 1, "pin_memory" : True} if use_cuda else {}
    
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotateDeformCrop(sigma=10, points=10, crop=crop_dim)])
    train_data = MaskDataset(args.data, "train", args.loss,
                             transform=train_trans)
    train_load = DataLoader(train_data, batch_size=args.batch_size, 
                            shuffle=True, **datakw)
    test_data = MaskDataset(args.data, "test", args.loss,
                            transform=transforms.RandomCrop(crop_dim))
    test_load = DataLoader(test_data, batch_size=args.batch_size,
                           shuffle=True, **datakw)
    
    # epoch loop
    min_test_loss = inf
    train_losses = []
    test_losses  = []
    for epoch in range(args.epochs):
        # training
        train_loss = train(train_load, net, crit, opt, epoch, device,
                           args.crop_size, args.print_freq)
        train_losses.append(train_loss)
        test_loss = test(test_load, net, crit, epoch, device,
                         args.crop_size, args.print_freq)
        test_losses.append(test_loss)
        if (test_loss < min_test_loss) and args.save_path is not None:
            save_checkpoint(os.path.join(args.save_path, "model.pth"),
                            net, opt, epoch)
            min_test_loss = test_loss
            print("saved checkpoint")

    # write losses to csv file
    if args.save_path is not None:
        with open(os.path.join(args.save_path, "losses.csv"), "w") as loss_csv:
            fieldnames = ["train", "test"]
            writer = csv.DictWriter(loss_csv, fieldnames=fieldnames)
            writer.writeheader()
            for tr, te in zip(train_losses, test_losses):
                writer.writerow({"train" : tr, "test" : te})
    
    return 0


def train(data, model, criterion, optimizer, epoch, device, output_size, 
          prog_disp=1):
    """per-epoch training loop
    """
    avgloss = AvgValueTracker("Loss", ":.4e")    # loss
    avgbtme = AvgValueTracker("Time", ":6.3f")  # batch time
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Epoch: [{}]".format(epoch))
    
    model.train()               # switch to train mode
    strt_time = time.time()
    for idx, (img, msk) in enumerate (data):
        msk = center_crop(msk, output_size)  # so that mask matches output size
        img = img.to(device)
        msk = msk.to(device)
        # computation
        out = model(img)
        loss = criterion(out, msk)
        # gradient/SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record
        avgloss.update(loss.item(), img.size(0))
        avgbtme.update(time.time() - strt_time)
        strt_time = time.time()
        # show progress
        if idx % prog_disp == 0:
            prog.display(idx)
    return avgloss.avg


def test(data, model, criterion, epoch, device, output_size, prog_disp=1):
    """per-epoch testing loop
    """
    avgloss = AvgValueTracker("Loss", ":.4e")
    avgbtme = AvgValueTracker("Time", ":6.3f")
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Test: ")
    
    model.eval()
    strt_time = time.time()
    with torch.no_grad():
        for idx, (img, msk) in enumerate(data):
            msk = center_crop(msk, output_size)  # so that mask matches output size
            img = img.to(device)
            msk = msk.to(device)
            # computation
            out = model(img)
            loss = criterion(out, msk)
            # record
            avgloss.update(loss.item(), img.size(0))
            avgbtme.update(time.time()-strt_time)
            strt_time = time.time()
            # show progress
            if idx % prog_disp == 0:
                prog.display(idx)
    return avgloss.avg


def reqd_input_dim(output_dim, depth, kernel_size=3):
    """Determine required input size to match desired output dimension
    if no padding used in UNet
    """
    orig_dim = output_dim
    for l in range(depth):
        # each layer has 2 convolutions
        for c in range(2):
            output_dim = _conv2d_output_size(output_dim, stride=1, padding=0,
                                             dilation=1, kernel_size=kernel_size)
    return output_dim


def _conv2d_output_size(dim_in, stride=1, padding=0,
                        dilation=1, kernel_size=3):
    return (dim_in + 2 * padding - dilation * (kernel_size - 1) - 1) / \
        stride + 1


def _convtranspose2d_output_size(dim_in, stride=2, padding=0, dilation=1,
                                 kernel_size=2, output_padding=0):
    return (dim_in - 1) * stride - 2 * padding + dilation * \
        (kernel_size - 1) + output_padding + 1

           
class AvgValueTracker(object):
    """Computes/stores average & current value
    
    NOTE: ripped from PyTorch imagenet example
    """
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
        
    
class ProgressShower(object):
    """Show progress of training
    
    NOTE: ripped from PyTorch imagenet example
    """
    def __init__(self, n_batch, trackers, prefix=""):
        self.fmt = self._get_batch_fmtstr(n_batch)
        self.trackers = trackers
        self.prefix = prefix
    
    def display(self, batch):
        entries = [self.prefix + self.fmt.format(batch)]
        entries += [str(t) for t in self.trackers]
        print("\t".join(entries))
    
    def _get_batch_fmtstr(self, n_batch):
        n_dig = len(str(n_batch // 1))
        fmt = "{:" + str(n_dig) + "d}"
        return "[" + fmt + "/" + fmt.format(n_batch) + "]"
    

def save_checkpoint(filepath, model, optimizer, epoch):
    """Save checkpoint to file
    """
    chckpt = {"model" : model, 
              "optimizer" : optimizer, 
              "epoch" : epoch}
    torch.save(chckpt, filepath)
    
    
if __name__ == "__main__":
    """command line UI
    """
    parser = argparse.ArgumentParser(description="UNet training script")
    # data/loss parameters
    parser.add_argument("-d", "--data", type=str, help="path to data folder")
    parser.add_argument("-l", "--loss", type=str, default="ce",
                        choices=["j3", "j4", "ce"],
                        help="type of loss function")
    parser.add_argument("-c", "--crop-size", type=int, default=256,
                        help="size of region to crop from original images")
    # training/optimization parameters
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("-bs", "--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="number of epochs")
    # unet parameters
    parser.add_argument("-ud", "--unet-depth", type=int, default=3,
                        help="depth of the UNet")
    parser.add_argument("-up", "--unet-pad", action="store_true", default=False,
                        help="use padding in UNet so input matches output size")
    parser.add_argument("-ub", "--unet-batchnorm", action="store_true", 
                        default=True,
                        help="use batch norm")
    parser.add_argument("-um", "--unet-upmode", type=str, default="upconv", 
                        choices=["upconv", "upsample"],
                        help="unet upsampling mode")
    # misc
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed")
    parser.add_argument("-pf", "--print-freq", type=int, default=1,
                        help="frequency to show progress during training")
    parser.add_argument("-nc", "--no-cuda", action="store_true",
                        default=False, help="disable CUDA")
    parser.add_argument("-sp", "--save-path", type=str, default=None,
                        help="directory to save training stats/models to")
    
    ec = main(**vars(parser.parse_args()))
    exit(ec)
