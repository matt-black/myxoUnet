"""Training script
"""

# python
import os
from math import inf
import time
import random
import argparse

# pytorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from unet import UNet
from data import MaskDataset

def main(**kwargs):
    args = argparse.Namespace(**kwargs)
    # random seed?
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    # use cuda?
    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # make unet/corresp. loss function
    if args.loss == "j3":
        n_class = 3
        raise NotImplementedError()
    elif args.loss == "j4" or args.loss == "w":
        n_class = 4
        raise NotImplementedError()
    else:
        n_class = 1
        crit = nn.CrossEntropyLoss()
    net = UNet(n_class, 2, 4, 6, True, True, 'upconv').to(device)

    # setup optimizer
    opt = optim.SGD(net.parameters(), args.learning_rate)
    
    # build up train/test datasets
    if not (os.path.isdir(args.data)):
        raise Exception("specified data directory doesn't exist")
    datakw = {"num_workers" : 1, "pin_memory" : True} if use_cuda else {}
    train_data = MaskDataset(args.data, "train", args.loss)
    train_load = DataLoader(train_data, batch_size=args.batch_size, 
                            shuffle=True, **datakw)
    test_data = MaskDataset(args.data, "test", args.loss)
    test_load = DataLoader(test_data, batch_size=args.batch_size,
                           shuffle=True, **datakw)
    
    # epoch loop
    min_test_loss = inf
    for epoch in range(args.epochs):
        # training
        train(train_load, net, crit, opt, epoch, device, args.print_freq)
        test_loss = test(test_load, net, crit, epoch, device, args.print_freq)
        if (test_loss < min_test_loss) and args.save_path is not None:
            save_checkpoint(args.save_path, net, opt, epoch)
            min_test_loss = test_loss
            print("saved checkpoint")
    
    return 0

def train(data, model, criterion, optimizer, epoch, device, prog_disp=1):
    """
    """
    avgloss = AvgValueTracker("Loss", ":.4e")    # loss
    avgbtme = AvgValueTracker("Time", ":6.3f")  # batch time
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Epoch: [{}]".format(epoch))
    
    model.train()               # switch to train mode
    end_time = time.time()
    for idx, (img, msk) in enumerate (data):
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
        avgbtme.update(time.time() - end_time)
        end_time = time.time()
        # show progress
        if idx % prog_disp == 0:
            prog.display(idx)


def test(data, model, criterion, epoch, device, prog_disp=1):
    """
    """
    avgloss = AvgValueTracker("Loss", ":.4e")
    avgbtme = AvgValueTracker("Time", ":6.3f")
    prog = ProgressShower(len(data), [avgloss, avgbtme], 
                          prefix="Test: ")
    
    model.eval()
    end_time = time.time()
    with torch.no_grad():
        for idx, (img, msk) in enumerate(data):
            img = img.to(device)
            msk = msk.to(device)
            # computation
            out = model(img)
            loss = criterion(out, msk)
            # record
            avgloss.update(loss.item(), img.size(0))
            avgbtme.update(time.time()-end_time)
            end_time = time.time()
            # show progress
            if idx % prog_disp == 0:
                prog.display(idx)
    return avgloss.avg

            
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
    parser.add_argument("-d", "--data", type=str, help="path to data folder")
    parser.add_argument("-l", "--loss", type=str, default="ce",
                        choices=["j3", "j4", "ce", "w"],
                        help="type of loss function")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("-bs", "--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="number of epochs")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed")
    parser.add_argument("-pf", "--print-freq", type=int, default=1,
                        help="frequency to show progress during training")
    parser.add_argument("-nc", "--no-cuda", action="store_true",
                        default=False, help="disable CUDA")
    parser.add_argument("-sp", "--save-path", type=str, default=None,
                        help="path to save checkpoint to")
    ec = main(**vars(parser.parse_args()))
    exit(ec)
