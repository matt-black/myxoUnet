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

from bunet import BUNet
from data import MaskDataset
from data import RandomRotateDeformCrop
from loss import BUNetLoss
from util import AvgValueTracker, ProgressShower


def main(**kwargs):
    args = argparse.Namespace(**kwargs)
    # make sure lr-scheduling is valid
    if len(args.reduce_lr_plateau) > 0 and len(args.step_lr) > 0:
        raise Exception("cant reduce on plateau and use step-lr")
    if len(args.reduce_lr_plateau) > 0 and args.no_test:
        raise Exception("cant reduce on plateau without a testing")
    
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
        fldr_name = "bunet_{d}".format(d=date.today().strftime("%Y-%m-%d"))
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
    net = BUNet(in_channels=1,
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
        out, _, _ = net(inp)
        out_sze = out.size(-1)
        padval = args.crop_size - out_sze
        # now try and see if it works
        inp = torch.rand(1, 1, args.crop_size+padval, 
                         args.crop_size+padval).to(device)
        out, _, _ = net(inp)
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
    if args.do_deform_transform:
        train_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotateDeformCrop(sigma=5, points=5, crop=crop_dim)])
    else:
        train_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-15,15)),
            transforms.RandomCrop(crop_dim)])
    
    train_data = MaskDataset(args.data, "train",
                             transform=train_trans,
                             stat_global=args.data_global_stats)
    train_load = DataLoader(train_data, batch_size=args.batch_size, 
                            shuffle=True, **datakw)
    if not args.no_test:
        test_trans = transforms.RandomCrop(crop_dim)
        
        test_data = MaskDataset(args.data, "test",
                                transform=test_trans,
                                stat_global=args.data_global_stats)
        test_load = DataLoader(test_data, batch_size=args.batch_size,
                               shuffle=True, **datakw)
    
    # make loss function
    crit = BUNetLoss(args.alpha, args.beta, args.gamma)
    
    # save input arguments to json file
    if args.save_path is not None:
        with open(os.path.join(args.save_path, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    # setup optimizer
    if args.sgd:
        opt = optim.SGD(net.parameters(), lr=args.learning_rate)
    else:
        opt = optim.Adam(net.parameters(), args.learning_rate)
    # setup learning rate scheduler
    if len(args.step_lr) == 0 and len(args.reduce_lr_plateau) == 0:
        scheduler = None
    elif len(args.step_lr) > 0:
        params = args.step_lr.split(",")
        scheduler = optim.lr_scheduler.StepLR(opt,
                                              step_size=int(params[0]),
                                              gamma=float(params[1]),
                                              verbose=True)
    else:
        params = args.reduce_lr_plateau.split(",")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                         mode='min',
                                                         factor=float(params[0]),
                                                         patience=int(params[1]),
                                                         threshold=float(params[2]),
                                                         threshold_mode='rel',
                                                         verbose=True)
        
    # epoch loop
    min_test_loss = inf
    train_losses = []; train_lfs = []; train_lbs = []; train_lrs = [];
    test_losses  = []; test_lfs = []; test_lbs = []; test_lrs = [];
    for epoch in range(args.epochs):
        # training
        train_loss, (train_lf, train_lb, train_lr) = train(
            train_load, net, crit, opt, epoch, device,
            args.crop_size, args.print_freq)
        train_losses.append(train_loss)
        train_lfs.append(train_lf)
        train_lbs.append(train_lb)
        train_lrs.append(train_lr)
        # do testing
        if args.no_test:
            test_loss = 0
            test_lf = 0; test_lb = 0; test_lr = 0;
        else:
            test_loss, (test_lf, test_lb, test_lr) = test(
                test_load, net, crit, epoch, device,
                args.crop_size, args.print_freq)
        test_losses.append(test_loss)
        test_lfs.append(test_lf)
        test_lbs.append(test_lb)
        test_lrs.append(test_lr)
        # do scheduling
        if scheduler is not None:
            if len(args.reduce_lr_plateau) > 0:  # ReduceLROnPlateau
                dyn_thresh = min_test_loss * (1 - scheduler.threshold)
                print("Dynamic Threshold: {:4e}".format(dyn_thresh))
                scheduler.step(test_loss)
            else:               # must be StepLR
                scheduler.step()
        
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
        with open(os.path.join(args.save_path, "train_loss.csv"), "w") as train_csv:
            fieldnames = ["epoch", "loss", "l_f", "l_b", "l_r"]
            writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
            writer.writeheader()
            for ep, (l, lf, lb, lr) in enumerate (
                    zip(train_losses, train_lfs, train_lbs, train_lrs)):
                writer.writerow({"epoch" : ep, "loss" : l,
                                 "l_f" : lf, "l_b" : lb, "l_r" : lr})
        if not args.no_test:
            with open(os.path.join(args.save_path, "test_loss.csv"), "w") as test_csv:
                fieldnames = ["epoch", "loss", "l_f", "l_b", "l_r"]
                writer = csv.DictWriter(test_csv, fieldnames=fieldnames)
                writer.writeheader()
                for ep, (l, lf, lb, lr) in enumerate (
                        zip(test_losses, test_lfs, test_lbs, test_lrs)):
                    writer.writerow({"epoch" : ep, "loss" : l,
                                     "l_f" : lf, "l_b" : lb, "l_r" : lr})
    return 0


def train(data, model, criterion, optimizer, epoch, device, output_size, 
          prog_disp=1):
    """per-epoch training loop
    """
    avgloss = AvgValueTracker("Loss", ":.4e")    # loss
    avgl_f = AvgValueTracker("L_fuse", ":4e")
    avgl_b = AvgValueTracker("L_bnd", ":4e")
    avgl_r = AvgValueTracker("L_reg", ":4e")
    avgbtme = AvgValueTracker("Time", ":6.3f")  # batch time
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Epoch: [{}]".format(epoch))
    
    model.train()               # switch to train mode
    strt_time = time.time()
    for idx, (img, mask, dist, bord) in enumerate (data):
        # center crop so predictions match net output
        mask = center_crop(mask, output_size)
        dist = center_crop(dist, output_size)
        bord = center_crop(bord, output_size)
        # send to appropriate device
        img = img.to(device)
        mask = mask.to(device)
        dist = dist.to(device)
        bord = bord.to(device)
        # computation
        fused, dist_pred, bord_pred = model(img)
        # for cross entropy predictions, take softmax
        fused = F.softmax(fused, dim=1)
        bord_pred = F.softmax (bord_pred, dim=1)
        # compute loss
        loss, (l_f, l_b, l_r) = criterion(fused, mask,
                                          dist_pred, dist,
                                          bord_pred, bord)
        # gradient/SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record
        avgloss.update(loss.item(), img.size(0))
        avgl_f.update(l_f.item(), img.size(0))
        avgl_b.update(l_b.item(), img.size(0))
        avgl_r.update(l_r.item(), img.size(0))
        avgbtme.update(time.time() - strt_time)
        strt_time = time.time()
        # show progress
        if idx % prog_disp == 0:
            prog.display(idx)
    return avgloss.avg, (avgl_f.avg, avgl_b.avg, avgl_r.avg)


def test(data, model, criterion, epoch, device, output_size, prog_disp=1):
    """per-epoch testing loop
    """
    avgloss = AvgValueTracker("Loss", ":.4e")
    avgl_f = AvgValueTracker("L_fuse", ":4e")
    avgl_b = AvgValueTracker("L_bnd", ":4e")
    avgl_r = AvgValueTracker("L_reg", ":4e")
    avgbtme = AvgValueTracker("Time", ":6.3f")
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Test: ")
    
    model.eval()
    strt_time = time.time()
    with torch.no_grad():
        for idx, (img, mask, dist, bord) in enumerate(data):
            # center crop so predictions match net output
            mask = center_crop(mask, output_size)
            dist = center_crop(dist, output_size)
            bord = center_crop(bord, output_size)
            # send to appropriate device
            img = img.to(device)
            mask = mask.to(device)
            dist = dist.to(device)
            bord = bord.to(device)
            # computation
            fused, dist_pred, bord_pred = model(img)
            # for cross entropy predictions, take softmax
            fused = F.softmax(fused, dim=1)
            bord_pred = F.softmax (bord_pred, dim=1)
            # compute loss
            loss, (l_f, l_b, l_r) = criterion(fused, mask,
                                              dist_pred, dist,
                                              bord_pred, bord)
            # record
            avgloss.update(loss.item(), img.size(0))
            avgl_f.update(l_f.item(), img.size(0))
            avgl_b.update(l_b.item(), img.size(0))
            avgl_r.update(l_r.item(), img.size(0))
            avgbtme.update(time.time()-strt_time)
            strt_time = time.time()
            # show progress
            if idx % prog_disp == 0:
                prog.display(idx)
    return avgloss.avg, (avgl_f.avg, avgl_b.avg, avgl_r.avg)


def save_checkpoint(filepath, model, optimizer, scheduler, epoch):
    """Save checkpoint to file
    """
    chckpt = {"model" : model.state_dict(), 
              "optimizer" : optimizer.state_dict(),
              "scheduler" : scheduler.state_dict() if scheduler is not None else None,
              "epoch" : epoch}
    torch.save(chckpt, filepath)

    
def load_checkpoint(filepath, argz):
    if torch.cuda.is_available():
        chkpt = torch.load(filepath)
    else:
        chkpt = torch.load(filepath, map_location=torch.device("cpu"))
        
    # initialize network/optimizer
    net = BUNet(in_channels=1,
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
    # handle the scheduler
    if len(argz.reduce_lr_plateau) > 0:
        params = args.reduce_lr_plateau.split(",")
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                   mode='min',
                                                   factor=float(params[0]),
                                                   patience=int(params[1]),
                                                   threshold=float(params[2]),
                                                   threshold_mode='rel',
                                                   verbose=True)
        sch.load_state_dict(chkpt["scheduler"])
    elif len(argz.step_lr) > 0:
        params = argz.step_lr.split(",")
        sch = optim.lr_scheduler.StepLR(opt,
                                        step_size=int(params[0]),
                                        gamma=float(params[1]),
                                        verbose=True)
        sch.load_state_dict(chkpt["scheduler"])
    else:
        sch = None
    return net, opt, sch


if __name__ == "__main__":
    """command line UI
    """
    parser = argparse.ArgumentParser(description="BUNet training script")
    # data parameters
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to data folder")
    parser.add_argument("-dgs", "--data-global-stats", action="store_true", default=False,
                        help="use global-statistics based normalization")
    parser.add_argument("-dd", "--do-deform-transform", action="store_true", default=False,
                        help="do elastic deformation for data augmentation")
    parser.add_argument("-cs", "--crop-size", type=int, default=256,
                        help="size of region to crop from original images")
    # loss parameters
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="alpha parameter for BUNetLoss")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="beta parameter for BUNetLoss")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="gamma parameter for BUNetLoss")
    # training/optimization parameters
    parser.add_argument("--sgd", action="store_true", default=False,
                        help="use SGD instead of Adam")
    parser.add_argument("--no-test", action="store_true", default=False,
                        help="skip test/evaluation at each epoch")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("-rlrp", "--reduce-lr-plateau", type=str, default="",
                        help="reduce learning rate when test loss plateaus, form:'factor,patience,threshold'")
    parser.add_argument("-slr", "--step-lr", type=str, default="",
                        help="step learning rate parameters of form 'step_size,gamma'")
    parser.add_argument("-bs", "--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="number of epochs")
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
                        choices=["conv","maxpool"],
                        help="unet downsampling mode")
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
