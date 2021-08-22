"""Training script for DCAN
"""

# python
import os
import csv
import json
from math import inf, log10
from datetime import date
import time
import random
import argparse

# pytorch
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dcan import DCAN
from data import DCANDataset
from data import RandomRotateDeformCrop
import loss
from util import AvgValueTracker, ProgressShower, one_hot

def main(**kwargs):
    args = argparse.Namespace(**kwargs)

    # random seed?
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    # setup output folder
    if args.save:
        # generate output path
        fldr_name = "dcan_{d}".format(d=date.today().strftime("%Y-%m-%d"))
        args.save_path = os.path.join(os.getcwd(), fldr_name)
    else:
        args.save_path = None
    if args.save_path is not None:
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        else:  # folder exists, add random int to end so we don't get overla
            args.save_path = args.save_path + "_" + \
                "{:d}".format(random.randint(1,1000))
            os.mkdir(args.save_path)
        
    # use cuda?
    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # setup DCAN
    print(args.dcan_upmode)
    net = DCAN(in_channels=1,
               depth=args.dcan_depth,
               wf=args.dcan_wf,
               kernel_size=args.dcan_kernel_size,
               padding=args.dcan_pad,
               batch_norm=args.dcan_batchnorm,
               up_mode=args.dcan_upmode,
               out_dim=args.dcan_outdim)
    net = net.to(device)
            
    # build up train/test datasets
    if not (os.path.isdir(args.data)):
        raise Exception("specified data directory doesn't exist")
    datakw = {"num_workers" : 1, "pin_memory" : True} if use_cuda else {}
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotateDeformCrop(sigma=5, points=5, crop=args.crop_size)])
    train_data = DCANDataset(args.data, "train",
                             transform=train_trans, 
                             stat_norm=args.data_statnorm)
    train_load = DataLoader(train_data, batch_size=args.batch_size, 
                            shuffle=True, **datakw)
    test_data = DCANDataset(args.data, "test",
                            transform=transforms.RandomCrop(args.crop_size),
                            stat_norm=args.data_statnorm)
    test_load = DataLoader(test_data, batch_size=args.batch_size,
                           shuffle=True, **datakw)
    
    # make loss function
    crit = loss.DcanLoss(eps=1e-6)
    
    # save input arguments to json file
    if args.save_path is not None:
        with open(os.path.join(args.save_path, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    # setup optimizer
    if args.sgd:
        opt = optim.SGD(net.parameters(), lr=args.learning_rate)
    else:
        opt = optim.Adam(net.parameters(), args.learning_rate)
    
    # epoch loop
    min_test_loss = inf
    train_losses = []
    test_losses  = []
    for epoch in range(args.epochs):
        # training
        wa = 10 ** (-log10(epoch+1)-1)
        wa = 0 if wa < 10 ** (-3) else wa
        train_loss = train(train_load, net, crit, opt, epoch, wa, 
                           args.dcan_outdim, device, 
                           prog_disp=args.print_freq)
        train_losses.append(train_loss)
        test_loss = test(test_load, net, crit, epoch, args.dcan_outdim, 
                         device, prog_disp=args.print_freq)
        test_losses.append(test_loss)
        if (test_loss < min_test_loss) and args.save_path is not None:
            save_checkpoint(os.path.join(args.save_path, "model.pth"),
                            net, opt, epoch)
            min_test_loss = test_loss
            print("saved checkpoint")

    # write losses to csv file
    if args.save_path is not None:
        with open(os.path.join(args.save_path, "losses.csv"), "w") as loss_csv:
            fieldnames = ["epoch", "train", "test"]
            writer = csv.DictWriter(loss_csv, fieldnames=fieldnames)
            writer.writeheader()
            for ep, (tr, te) in enumerate (zip(train_losses, test_losses)):
                writer.writerow({"epoch" : ep, "train" : tr, "test" : te})
    
    return 0


def train(data, model, criterion, optimizer, epoch, wa, outdim, device, 
          prog_disp=1):
    """per-epoch training loop
    """
    avgloss = AvgValueTracker("Loss", ":.4e")    # loss
    avgbtme = AvgValueTracker("Time", ":6.3f")  # batch time
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Epoch: [{}]".format(epoch))
    
    model.train()               # switch to train mode
    strt_time = time.time()
    for idx, (img, cell_msk, cntr_msk) in enumerate (data):
        img = img.to(device)
        # crop masks to match output size of network
        cell_msk = transforms.functional.center_crop(cell_msk, [outdim, outdim])
        cntr_msk = transforms.functional.center_crop(cntr_msk, [outdim, outdim])
        cell_msk = one_hot(cell_msk, 2, device, dtype=torch.float32)
        cntr_msk = one_hot(cntr_msk, 2, device, dtype=torch.float32)
        # computation
        m0, mc, c123 = model(img)
        # convert outputs to probability maps
        p0 = F.softmax(m0, dim=1)
        pc = F.softmax(mc, dim=1)
        # intermediate masks
        p123 = [(F.softmax(i0, dim=1), F.softmax(ic, dim=1)) 
                for (i0, ic) in c123]
        
        loss = criterion(p0, cell_msk, pc, cntr_msk)
        l123 = [wa*criterion(i0, cell_msk, ic, cntr_msk) for (i0, ic) in p123]
        total_loss = loss + torch.sum(torch.stack(l123, dim=0))
        # gradient/SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # record
        avgloss.update(total_loss.item(), img.size(0))
        avgbtme.update(time.time() - strt_time)
        strt_time = time.time()
        # show progress
        if idx % prog_disp == 0:
            prog.display(idx)
    return avgloss.avg


def test(data, model, criterion, epoch, outdim, device, prog_disp=1):
    """per-epoch testing loop
    """
    avgloss = AvgValueTracker("Loss", ":.4e")
    avgbtme = AvgValueTracker("Time", ":6.3f")
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Test: ")
    
    model.eval()
    strt_time = time.time()
    with torch.no_grad():
        for idx, (img, cell_msk, cntr_msk) in enumerate(data):
            img = img.to(device)
            cell_msk = transforms.functional.center_crop(cell_msk, 
                                                         [outdim, outdim])
            cntr_msk = transforms.functional.center_crop(cntr_msk, 
                                                         [outdim, outdim])
            cell_msk = one_hot(cell_msk, 2, device, dtype=torch.float32)
            cntr_msk = one_hot(cntr_msk, 2, device, dtype=torch.float32)
            # computation
            m0, mc, _ = model(img)
            # convert outputs to probability maps
            p0 = F.softmax(m0, dim=1)
            pc = F.softmax(mc, dim=1)
            loss = criterion(p0, cell_msk, pc, cntr_msk)
            # record
            avgloss.update(loss.item(), img.size(0))
            avgbtme.update(time.time()-strt_time)
            strt_time = time.time()
            # show progress
            if idx % prog_disp == 0:
                prog.display(idx)
    return avgloss.avg


def save_checkpoint(filepath, model, optimizer, epoch):
    """Save checkpoint to file
    """
    chckpt = {"model" : model.state_dict(), 
              "optimizer" : optimizer.state_dict(), 
              "epoch" : epoch}
    torch.save(chckpt, filepath)

    
def load_checkpoint(filepath, argz):
    if torch.cuda.is_available():
        chkpt = torch.load(filepath)
    else:
        chkpt = torch.load(filepath, map_location=torch.device("cpu"))
        
    # initialize network/optimizer
    net = DCAN(in_channels=1,
               depth=argz.dcan_depth,
               wf=argz.dcan_wf,
               kernel_size=argz.dcan_kernel_size,
               padding=argz.dcan_pad,
               batch_norm=argz.dcan_batchnorm,
               up_mode=argz.dcan_upmode,
               out_dim=argz.dcan_outdim)
    if argz.sgd:
        opt = optim.SGD(net.parameters(), lr=argz.learning_rate)
    else:
        opt = optim.Adam(net.parameters(), lr=argz.learning_rate)
    # load params from checkpoint
    net.load_state_dict(chkpt["model"])
    opt.load_state_dict(chkpt["optimizer"])
    return net, opt

            
if __name__ == "__main__":
    """command line UI
    """
    parser = argparse.ArgumentParser(description="DCAN training script")
    # data/loss parameters
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to data folder")
    parser.add_argument("-cs", "--crop-size", type=int, default=280,
                        help="size of region to crop from original images")
    parser.add_argument("-ds", "--data-statnorm",
                        action="store_true", default=True,
                        help="normalize images by mean/std instead of just [0,1]")
    parser.add_argument("--sgd", action="store_true", default=False,
                        help="use SGD instead of Adam")
    # training/optimization parameters
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("-bs", "--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="number of epochs")
    # unet parameters
    parser.add_argument("-dd", "--dcan-depth", type=int, default=5,
                        help="network depth")
    parser.add_argument("-df", "--dcan-wf", type=int, default=4,
                        help="log2 number of filters in first layer")
    parser.add_argument("-dk", "--dcan-kernel-size", type=int, default=3,
                        help="kernel size of downsampling layers")
    parser.add_argument("-dp", "--dcan-pad", action="store_true", default=False,
                        help="use padding in DCAN downsampling layers")
    parser.add_argument("-db", "--dcan-batchnorm", action="store_true", 
                        default=True, help="use batch norm")
    parser.add_argument("-dm", "--dcan-upmode", type=str, default="upconv",
                        choices=["upconv", "upsample"],
                        help="upsampling mode of network")
    parser.add_argument("-do", "--dcan-outdim", type=int, default=256,
                        help="output size of DCAN")
    # misc
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed")
    parser.add_argument("-pf", "--print-freq", type=int, default=1,
                        help="frequency to show progress during training")
    parser.add_argument("-nc", "--no-cuda", action="store_true",
                        default=False, help="disable CUDA")
    parser.add_argument("-sv", "--save", action="store_true",
                        help="save training stats/models to directory")
    
    ec = main(**vars(parser.parse_args()))
    exit(ec)
