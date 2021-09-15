"""Training script for MaskRCNN
"""
import os, sys
import csv
import json
from datetime import date
import random
import argparse

sys.path.insert(1, os.path.abspath("."))
sys.path.insert(1, os.path.abspath("./pytv"))
from pytv.engine import train_one_epoch, evaluate
import pytv.utils as utils

from data import MaskRCNNDataset

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def make_new_maskr(n_class=2, hidden_layer=256, box_detect=300,
                   backbone_train=False):
    """generate new MaskRCNN model from pretrained ResNet
    """
    if backbone_train:
        n_train_back = 1
    else:
        n_train_back = None
    # load pretrained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True, progress=True, trainable_backbone_layers=n_train_back,
        box_detections_per_img=box_detect)

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
        data_kw = {"num_workers" : 4, "pin_memory" : True}
    else:
        device = torch.device('cpu')
        data_kw = {"num_workers" : 1}
    # setup dataset/loader
    train_trans = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(degrees=(-15,15)),
        torchvision.transforms.RandomCrop(args.crop_size)])
    dataset_train = MaskRCNNDataset(args.data, "train", train_trans, False)
    data_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=utils.collate_fn, **data_kw)
    
    test_trans = torchvision.transforms.CenterCrop(args.crop_size)
    dataset_test = MaskRCNNDataset(args.data, "test", test_trans, False)
    data_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        collate_fn=utils.collate_fn, **data_kw)

    # generate model
    net = make_new_maskr(n_class=2,
                         hidden_layer=args.hidden_layer,
                         box_detect=args.box_detections_per_img,
                         backbone_train=args.train_backbone)
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
        metlog = train_one_epoch(net, optim, data_train, device, epoch,
                                 print_freq=1)
        if args.save_path is not None:
            if epoch == 0:      # create file, write header
                with open(os.path.join(args.save_path, "losses.csv"), "w") as f:
                    fieldnames = [k for k in metlog.meters.keys()]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
            # write current losses to losses.csv file
            with open(os.path.join(args.save_path, "losses.csv"), "a+") as f:
                fieldnames = [k for k in metlog.meters.keys()]
                row = {k : v.value for k, v in
                       zip(metlog.meters.keys(), metlog.meters.values())}
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
        # step the lr scheduler
        lr_scheduler.step()

        # do evaluation on test dataset
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
    # data parameters
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to data folder")
    parser.add_argument("-cs", "--crop-size", type=int, default=256,
                        help="dimension of image crops")
    # training parameters
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="# of training epochs")
    parser.add_argument("-bs", "--batch-size", type=int, default=2,
                        help="batch size for training")
    # network parameters
    parser.add_argument("-hl", "--hidden-layer", type=int, default=256,
                        help="size of hidden layer")
    parser.add_argument("-bd", "--box-detections-per-img", type=int, default=300,
                        help="# of box detections per image")
    parser.add_argument("-tb", "--train-backbone", action="store_true", default=False,
                        help="enable training of final backbone layer")
    # misc.
    parser.add_argument("-s", "--save", action="store_true", default=False,
                        help="save final model")    
    ec = main(**vars(parser.parse_args()))
    exit(ec)
    
