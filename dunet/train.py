"""Training script
"""

# python
import os, pwd
import csv
import json
from math import inf
from datetime import date
import time
import random
import argparse

# pytorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import center_crop

from dunet import DUnet
from loss import DUnetSmoothL1
from data import MaskDataset, SizeScaledMaskDataset
from data import AddGaussianNoise, RandomContrastAdjust
from util import AvgValueTracker, ProgressShower


def main(**kwargs):
    args = argparse.Namespace(**kwargs)

    # make sure arg combination is valid
    if args.no_test and args.schedule_learning_rate:
        raise Exception("can't schedule learning rate without testing")
    
    # random seed?
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # use cuda?
    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
        
    # setup output folder
    if args.save:
        # generate output path
        fldr_name = "dunet_{d}".format(d=date.today().strftime("%Y-%m-%d"))
        if use_cuda: #cuda available, assume we're on cluster
            save_base = os.path.join('/', 'scratch', 'gpfs',
                                     pwd.getpwuid(os.getuid()).pw_name)
        else:
            save_base = os.getcwd()
        args.save_path = os.path.join(save_base, fldr_name)
    else:
        args.save_path = None
    
    if args.save_path is not None:
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        else:  # folder exists, add random int to end so we don't get overla
            args.save_path = args.save_path + "_" + \
                "{:d}".format(random.randint(1,1000))
            os.mkdir(args.save_path)
   
    # setup UNet
    net = DUnet(in_channels=1,
                depth=args.unet_depth,
                wf=args.unet_wf,
                padding=args.unet_pad,
                batch_norm=(not args.no_batchnorm),
                up_mode=args.unet_upmode,
                down_mode=args.unet_downmode)
    net = net.to(device)
    
    # TODO: fix this so you actually know what size to pad with instead of
    # just guessing until you find a good one
    if not args.unet_pad:
        inp = torch.rand(1, 1, args.crop_size, args.crop_size).to(device)
        out, _ = net(inp)
        out_sze = out.size(-1)
        padval = args.crop_size - out_sze
        # now try and see if it works
        inp = torch.rand(1, 1, args.crop_size+padval, 
                         args.crop_size+padval).to(device)
        out, _ = net(inp)
        out_sze = out.size(-1)
        # if sizes dont match, tell user to try again
        if out_sze != args.crop_size:
            raise ValueError(
                "crop size isnt valid with specified network structure")
        # if we got here, then the padding is ok
        crop_dim = args.crop_size + padval
        args.input_pad = padval
    else:
        crop_dim = args.crop_size
        args.input_pad = 0
        
    # build up train/test datasets
    if not (os.path.isdir(args.data)):
        raise Exception("specified data directory doesn't exist")
    datakw = {"num_workers" : 1, "pin_memory" : True} if use_cuda else {}
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.75),
        transforms.RandomVerticalFlip(p=0.75),
        transforms.RandomCrop(crop_dim), 
        transforms.RandomApply([transforms.RandomRotation(degrees=(-45,45))],
                               p=0.3)])
    train_img_trans = transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomApply([AddGaussianNoise(0, 0.5)], p=0.3)])
    
    if args.data_size_scale:
        train_data = SizeScaledMaskDataset(args.data, "train",
                                           crop_dim=args.crop_dim,
                                           transform=train_trans,
                                           img_transform=train_img_trans)
    else:
        train_data = MaskDataset(args.data, "train",
                                 transform=train_trans,
                                 img_transform=train_img_trans)
    train_load = DataLoader(train_data, batch_size=args.batch_size, 
                            shuffle=True, **datakw)
    if not args.no_test:        # do testing
        test_trans = transforms.RandomCrop(crop_dim)
        if args.data_size_scale:
            test_data = SizeScaledMaskDataset(args.data, "test",
                                              crop_dim=args.crop_dim,
                                              transform=test_trans,
                                              img_transform=None)
        else:
            test_data = MaskDataset(args.data, "test",
                                     transform=test_trans,
                                     img_transform=None)
        test_load = DataLoader(test_data, batch_size=args.batch_size,
                               shuffle=False, **datakw)
    
    # save input arguments to json file
    if args.save_path is not None:
        with open(os.path.join(args.save_path, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    # setup optimizer
    if args.sgd:
        opt = optim.SGD(net.parameters(), lr=args.learning_rate)
    else:
        opt = optim.Adam(net.parameters(), args.learning_rate)
    if args.schedule_learning_rate:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.25, patience=12, min_lr=6e-5,
            verbose=None)
    else:
        scheduler = None
    # epoch loop
    min_test_loss = inf
    train_losses = []
    train_cdl = []
    train_ndl = []
    test_losses  = []
    test_cdl = []
    test_ndl = []
    for epoch in range(args.epochs):
        # training
        train_loss, train_cd, train_nd = train(
            train_load, net, opt, epoch, device,
            args.crop_size, args.print_freq)
        # record losses
        train_losses.append(train_loss)
        train_cdl.append(train_cd)
        train_ndl.append(train_nd)
        if args.no_test:
            test_loss = 0
            test_cd = 0
            test_nd = 0
        else:
            test_loss, test_cd, test_nd = test(
                test_load, net, epoch, device,
                args.crop_size, args.print_freq)
            if scheduler is not None:
                scheduler.step(test_loss)
        test_losses.append(test_loss)
        test_cdl.append(test_cd)
        test_ndl.append(test_nd)
        if (test_loss < min_test_loss) and args.save_path is not None:
            if args.no_test:
                continue
            save_checkpoint(os.path.join(args.save_path, "model.pth"),
                            net, opt, scheduler, epoch)
            min_test_loss = test_loss
            print("saved checkpoint")
    
    if args.no_test:  # just save final model
        save_checkpoint(os.path.join(args.save_path, "model.pth"),
                        net, opt, scheduler, args.epochs-1)
    
    # write losses to csv file
    if args.save_path is not None:
        with open(os.path.join(args.save_path, "losses.csv"), "w") as loss_csv:
            fieldnames = ["epoch",
                          "train", "train_cd", "train_nd",
                          "test", "test_cd", "test_nd"]
            writer = csv.DictWriter(loss_csv, fieldnames=fieldnames)
            writer.writeheader()
            loss_zip = zip(train_losses, train_cdl, train_ndl,
                           test_losses, test_cdl, test_ndl)
            for ep, (tr, trc, trn, te, tec, ten) in enumerate (loss_zip):
                writer.writerow({"epoch" : ep,
                                 "train" : tr,
                                 "train_cd" : trc,
                                 "train_nd" : trn,
                                 "test" : te,
                                 "test_cd" : tec,
                                 "test_nd" : ten})
    return 0


def train(data, model, optimizer, epoch, device, output_size, 
          prog_disp=1):
    """per-epoch training loop
    """
    avgloss = AvgValueTracker("Loss", ":.4e")    # loss
    avgcd = AvgValueTracker("CD Loss", ":.4e")
    avgnd = AvgValueTracker("ND Loss", ":.4e")
    avgbtme = AvgValueTracker("Time", ":6.3f")  # batch time
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Epoch: [{}]".format(epoch))

    crit = nn.SmoothL1Loss(reduction="mean")
    model.train()               # switch to train mode
    strt_time = time.time()
    for idx, (img, d_cell, d_neig) in enumerate (data):
        # so that distances matches output size
        d_cell = center_crop(d_cell, output_size)  
        d_neig = center_crop(d_neig, output_size)
        img = img.to(device)
        d_cell = d_cell.to(device)
        d_neig = d_neig.to(device)
        # computation
        p_cell, p_neig = model(img)
        # compute loss
        l_cell = crit(p_cell, d_cell)
        l_neig = crit(p_neig, d_neig)
        loss = l_neig + l_cell
        # gradient/SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record
        avgloss.update(loss.item(), img.size(0))
        avgcd.update(l_cell.item(), img.size(0))
        avgnd.update(l_neig.item(), img.size(0))
        avgbtme.update(time.time() - strt_time)
        strt_time = time.time()
        # show progress
        if idx % prog_disp == 0:
            prog.display(idx)
    return avgloss.avg, avgcd.avg, avgnd.avg


def test(data, model, epoch, device, output_size, prog_disp=1):
    """per-epoch testing loop
    """
    avgloss = AvgValueTracker("Loss", ":.4e")
    avgcd = AvgValueTracker("CD Loss", ":.4e")
    avgnd = AvgValueTracker("ND Loss", ":.4e")
    avgbtme = AvgValueTracker("Time", ":6.3f")
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Test: ")
    crit = nn.SmoothL1Loss()
    model.eval()
    strt_time = time.time()
    with torch.no_grad():
        for idx, (img, d_cell, d_neig) in enumerate (data):
            # so that distances matches output size
            d_cell = center_crop(d_cell, output_size)  
            d_neig = center_crop(d_neig, output_size)
            img = img.to(device)
            d_cell = d_cell.to(device)
            d_neig = d_neig.to(device)
            # computation
            p_cell, p_neig = model(img)
            # compute loss
            l_cell = crit(p_cell, d_cell)
            l_neig = crit(p_neig, d_neig)
            loss = l_cell + l_neig
            # record
            avgloss.update(loss.item(), img.size(0))
            avgcd.update(l_cell.item(), img.size(0))
            avgnd.update(l_neig.item(), img.size(0))
            avgbtme.update(time.time() - strt_time)
            strt_time = time.time()
            # show progress
            if idx % prog_disp == 0:
                prog.display(idx)
    return avgloss.avg, avgcd.avg, avgnd.avg


def save_checkpoint(filepath, model, optimizer, scheduler, epoch):
    """Save checkpoint to file
    """
    if scheduler is None:
        sched_dict = None
    else:
        sched_dict = scheduler.state_dict()
    chckpt = {"model" : model.state_dict(), 
              "optimizer" : optimizer.state_dict(), 
              "epoch" : epoch,
              "scheduler" : sched_dict}
    torch.save(chckpt, filepath)

    
def load_checkpoint(filepath, argz):
    if torch.cuda.is_available():
        chkpt = torch.load(filepath)
    else:
        chkpt = torch.load(filepath, map_location=torch.device("cpu"))
        
    # initialize network/optimizer
    net = DUnet(in_channels=1,
                depth=argz.unet_depth,
                wf=argz.unet_wf,
                padding=argz.unet_pad,
                batch_norm=(not argz.no_batchnorm),
                up_mode=argz.unet_upmode,
                down_mode=argz.unet_downmode)
    if argz.sgd:
        opt = optim.SGD(net.parameters(), lr=argz.learning_rate)
    else:
        opt = optim.Adam(net.parameters(), lr=argz.learning_rate)
    # load params from checkpoint
    net.load_state_dict(chkpt["model"])
    opt.load_state_dict(chkpt["optimizer"])
    if chkpt["scheduler"] is not None:
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                   factor=0.25, patience=12,
                                                   min_lr=6e-5, verbose=True)
        sch.load_state_dict(chkpt["scheduler"])
    else:
        sch = None
    return net, opt, sch


if __name__ == "__main__":
    """command line UI
    """
    parser = argparse.ArgumentParser(description="Distance UNet training script")
    # data/loss parameters
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to data folder")
    parser.add_argument("-dss", "--data-size-scale", action="store_true", default=False,
                        help="use size-scaled datasets")
    parser.add_argument("-cs", "--crop-size", type=int, default=256,
                        help="size of region to crop from original images")
    parser.add_argument("--sgd", action="store_true", default=False,
                        help="use SGD instead of Adam")
    parser.add_argument("--no-test", action="store_true", default=False,
                        help="skip test/evaluation at each epoch")
    # training/optimization parameters
    parser.add_argument("-lr", "--learning-rate", type=float, default=8e-4,
                        help="learning rate")
    parser.add_argument("-bs", "--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="number of epochs")
    parser.add_argument("-slr", "--schedule-learning-rate", action="store_true",
                        default=False, help="do learning rate scheduling")
    # unet parameters
    parser.add_argument("-ud", "--unet-depth", type=int, default=3,
                        help="depth of the UNet")
    parser.add_argument("-uf", "--unet-wf", type=int, default=4,
                        help="log2 number of filters in first layer")
    parser.add_argument("-up", "--unet-pad", action="store_true", default=False,
                        help="use padding in UNet so input matches output size")
    parser.add_argument("-ub", "--no-batchnorm", action="store_true", 
                        default=False, help="use batch norm")
    parser.add_argument("-um", "--unet-upmode", type=str, default="upconv", 
                        choices=["upconv", "upsample"],
                        help="unet upsampling mode")
    parser.add_argument("-udm", "--unet-downmode", type=str, default="maxpool",
                        choices=["maxpool","conv"], help="unet downsampling mode")
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
