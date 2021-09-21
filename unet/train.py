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

from unet import UNet
from data import MaskDataset, SizeScaledMaskDataset
from data import RandomRotateDeformCrop
import loss
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
        fldr_name = "{d}_{l}".format(d=date.today().strftime("%Y-%m-%d"),
                                     l=args.loss)
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
    net = UNet(in_channels=1,
               n_classes=args.num_classes,
               depth=args.unet_depth,
               wf=args.unet_wf,
               padding=args.unet_pad,
               batch_norm=args.unet_batchnorm,
               up_mode=args.unet_upmode)
    net = net.to(device)
    
    # TODO: fix this so you actually know what size to pad with instead of
    # just guessing until you find a good one
    if not args.unet_pad:
        inp = torch.rand(1, 1, args.crop_size, args.crop_size).to(device)
        out_sze = net(inp).size(-1)
        padval = args.crop_size - out_sze
        # now try and see if it works
        inp = torch.rand(1, 1, args.crop_size+padval, 
                         args.crop_size+padval).to(device)
        out_sze = net(inp).size(-1)
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
    
    if args.data_size_scale:
        train_data = SizeScaledMaskDataset(args.data, "train",
                                           n_classes=args.num_classes,
                                           crop_dim=args.crop_dim,
                                           transform=train_trans,
                                           stat_global=args.data_global_stats)
    else:
        train_data = MaskDataset(args.data, "train",
                                 n_classes=args.num_classes,
                                 transform=train_trans,
                                 stat_global=args.data_global_stats)
    train_load = DataLoader(train_data, batch_size=args.batch_size, 
                            shuffle=True, **datakw)
    if not args.no_test:
        test_trans = transforms.RandomCrop(crop_dim)
        if args.data_size_scale:
            test_data = SizeScaledMaskDataset(args.data, "test",
                                              n_classes=args.num_classes,
                                              crop_dim=args.crop_dim,
                                              transform=test_trans,
                                              stat_global=args.data_global_stats)
        else:
            test_data = MaskDataset(args.data, "test",
                                     n_classes=args.num_classes,
                                     transform=test_trans,
                                     stat_global=args.data_global_stats)
        test_load = DataLoader(test_data, batch_size=args.batch_size,
                               shuffle=True, **datakw)
    
    # make loss function
    if args.loss == "jrce":
        crit = loss.JRegularizedCrossEntropyLoss()
    elif args.loss == "jrcew":
        # weight by class imbalance
        pct = train_data.class_percents()
        class_wgt = (1.0 - pct/100).to(device)
        crit = loss.JRegularizedCrossEntropyLoss(None, class_wgt)
    elif args.loss == "ce":
        crit = nn.CrossEntropyLoss()
    elif args.loss == "wce":
        # weight by class imbalance
        pct = train_data.class_percents()
        class_wgt = (1.0 - pct/100).to(device)
        crit = nn.CrossEntropyLoss(class_wgt)
    elif args.loss == "dice":
        crit = loss.DiceLoss()
    elif args.loss == "dsc":
        crit = loss.DiceRegularizedCrossEntropy()
    elif args.loss == "dssim":
        crit = loss.DSSIMLoss(11, reduction='mean', max_val=train_data.maxval)
    else:
        raise ValueError("invalid loss")
    
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
    train_losses = []
    test_losses  = []
    for epoch in range(args.epochs):
        # training
        train_loss = train(train_load, net, crit, opt, epoch, device,
                           args.crop_size, args.print_freq)
        train_losses.append(train_loss)
        # do testing
        if args.no_test:
            test_loss = 0
        else:
            test_loss = test(test_load, net, crit, epoch, device,
                             args.crop_size, args.print_freq)
        test_losses.append(test_loss)
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
        with open(os.path.join(args.save_path, "losses.csv"), "w") as loss_csv:
            fieldnames = ["epoch", "train", "test"]
            writer = csv.DictWriter(loss_csv, fieldnames=fieldnames)
            writer.writeheader()
            for ep, (tr, te) in enumerate (zip(train_losses, test_losses)):
                writer.writerow({"epoch" : ep, "train" : tr, "test" : te})
    
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
        out = F.softmax(out, dim=1)
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
            out = F.softmax(out, dim=1)
            loss = criterion(out, msk)
            # record
            avgloss.update(loss.item(), img.size(0))
            avgbtme.update(time.time()-strt_time)
            strt_time = time.time()
            # show progress
            if idx % prog_disp == 0:
                prog.display(idx)
    return avgloss.avg


def save_checkpoint(filepath, model, optimizer, scheduler, epoch):
    """Save checkpoint to file
    """
    chckpt = {"model" : model.state_dict(), 
              "optimizer" : optimizer.state_dict(),
              "scheduler" : scheduler.state_dict(),
              "epoch" : epoch}
    torch.save(chckpt, filepath)

    
def load_checkpoint(filepath, argz):
    if torch.cuda.is_available():
        chkpt = torch.load(filepath)
    else:
        chkpt = torch.load(filepath, map_location=torch.device("cpu"))
        
    # initialize network/optimizer
    net = UNet(in_channels=1,
               n_classes=argz.num_classes,
               depth=argz.unet_depth,
               wf=argz.unet_wf,
               padding=argz.unet_pad,
               batch_norm=argz.unet_batchnorm,
               up_mode=argz.unet_upmode)
    if argz.sgd:
        opt = optim.SGD(net.parameters(), lr=argz.learning_rate)
    else:
        opt = optim.Adam(net.parameters(), lr=argz.learning_rate)
    # load params from checkpoint
    net.load_state_dict(chkpt["model"])
    opt.load_state_dict(chkpt["optimizer"])
    # handle the scheduler
    if len(argz.reduce_lr_plateau) > 0:
        params = argz.reduce_lr_plateau.split(",")
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
    parser = argparse.ArgumentParser(description="UNet training script")
    # data/loss parameters
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to data folder")
    parser.add_argument("-dss", "--data-size-scale", action="store_true", default=False,
                        help="use size-scaled datasets")
    parser.add_argument("-dgs", "--data-global-stats", action="store_true", default=False,
                        help="use global-statistics based normalization")
    parser.add_argument("-dd", "--do-deform-transform", action="store_true", default=False,
                        help="do elastic deformation for data augmentation")
    parser.add_argument("-l", "--loss", type=str, default="ce",
                        help="type of loss function")
    parser.add_argument("-c", "--num-classes", type=int, default=2,
                        choices=[2,3,4,5], help="number of semantic classes")
    parser.add_argument("-cs", "--crop-size", type=int, default=256,
                        help="size of region to crop from original images")
    parser.add_argument("--sgd", action="store_true", default=False,
                        help="use SGD instead of Adam")
    parser.add_argument("--no-test", action="store_true", default=False,
                        help="skip test/evaluation at each epoch")
    # training/optimization parameters
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
    parser.add_argument("-sv", "--save", action="store_true",
                        help="save training stats/models to directory")
    
    ec = main(**vars(parser.parse_args()))
    exit(ec)
