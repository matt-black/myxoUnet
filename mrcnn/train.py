"""Training script for MaskRCNN
"""
import os, sys
import argparse
import json
from datetime import date
import random

sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath("./pytv"))
from pytv.engine import train_one_epoch, evaluate
import pytv.utils as utils

from data import MaskRCNNDataset

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def make_new_maskr(n_class=2, hidden_layer=256):
    """generate new MaskRCNN model from pretrained ResNet
    """
    # load pretrained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # replace box predictor head with new one (to be trained)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, n_class)

    # replace mask predictor with new one (to be trained)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask,
                                                       hidden_layer,
                                                       n_class)
    return model


def main(**kwargs):
    args = argparse.Namespace(**kwargs)

    # setup output folder
    if args.save:
        # generate output path
        fldr_name = "mrcnn_{d}".format(d=date.today().strftime("%Y-%m-%d"))
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

    # save input arguments to json file
    if args.save_path is not None:
        with open(os.path.join(args.save_path, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    # setup gpu/cpu device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # setup dataset/loader
    dataset_train = MaskRCNNDataset(args.data, "train", None)
    dataset_test = MaskRCNNDataset(args.data, "test", None)
    data_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True,
        num_workers=1, collate_fn=utils.collate_fn)
    data_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        num_workers=1, collate_fn=utils.collate_fn)

    # generate model
    net = make_new_maskr(n_class=2,
                         hidden_layer=256)
    net = net.to(device)

    # setup optimization with lr scheduling
    params = [p for p in net.parameters()
              if p.requires_grad]
    optim = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(args.epochs):
        train_one_epoch(net, optim, data_train, device, epoch,
                        print_freq=1)
        lr_scheduler.step()
        evaluate(net, data_test, device=device)

    if args.save:
        save_checkpoint(os.path.join(args.save_path, "model.pth"),
                        net, optim, lr_scheduler, args.epochs)

        
def save_checkpoint(filepath, model, optimizer, lr_scheduler,
                    epoch, n_class=2, hidden_layer=256):
    chkpt = {"model" : model.state_dict(),
              "optimizer" : optimizer.state_dict(),
              "epoch" : epoch,
              "lr_scheduler" : lr_scheduler.state_dict(),
              "n_class" : n_class,
              "hidden_layer" : hidden_layer}
    torch.save(chkpt, filepath)
    

def load_checkpoint(filepath, argz):
    if torch.cuda.is_available():
        chkpt = torch.load(filepath)
    else:
        chkpt = torch.load(filepath, map_location=torch.device("cpu"))
        
    # initialize network/optimizer
    net = make_new_maskr(chkpt["n_class"],
                         chkpt["hidden_layer"])
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005)
    
    lr_sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=3, gamma=0.1)
    
    # load params from checkpoint
    net.load_state_dict(chkpt["model"])
    opt.load_state_dict(chkpt["optimizer"])
    lr_sched.load_state_dict(chkpt['lr_scheduler'])

    return net, opt, lr_sched

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskRCNN training script")
    # data/loss parameters
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to data folder")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="# of training epochs")
    parser.add_argument("-s", "--save", action="store_true", default=False,
                        help="save final model")
    
    ec = main(**vars(parser.parse_args()))
    exit(ec)
    
