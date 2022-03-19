#!/usr/bin/env python
from torch.utils.data import Dataset
from glob import glob
from data.transforms import *
import cv2
import numpy as np
import random
from data.crowd_dataset import Crop_Dataset
import timm
from model.counTR import CounTR
import torch
import torch.nn as nn
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from train_loop import *
from utils import collate_fn
import os
import argparse

import warnings
warnings.filterwarnings("ignore")
torch.hub.set_dir('/data1/xluap/torch/hub')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='counTR, transformer based crowd counting model'
    )

    parser.add_argument(
        '--input-path',
        dest="path",
        type=str,
        help="The root path of the data directory"
    )
    
    parser.add_argument(
        '--train-image-folder',
        dest="train_image_folder",
        type=str,
        help="The directory name of the train image data"
    )
    
    parser.add_argument(
        '--train-dmap-folder',
        dest="train_dmap_folder",
        type=str,
        help="The directory name of the train density maps"
    )
    
    parser.add_argument(
        '--valid-image-folder',
        dest="valid_image_folder",
        type=str,
        help="The directory name of the valid image data"
    )
    
    parser.add_argument(
        '--valid-dmap-folder',
        dest="valid_dmap_folder",
        type=str,
        help="The directory name of the valid density maps"
    )
    
    parser.add_argument(
        '--output-folder',
        dest="folder",
        type=str,
        help="output folder for training"
    )
    
    parser.add_argument(
        '--crop_size',
        dest="crop_size",
        type=int,
        help="crop_size for train image",
        default=448
    )
    
    parser.add_argument(
        '--ext',
        dest="ext",
        type=str,
        help="The extension of input image",
        default='.jpg'
    )
    
    parser.add_argument(
        '--dmap-ext',
        dest="dmap_ext",
        type=str,
        help="The extension of density map",
        default='.npy'
    )
    
    parser.add_argument(
        '--device',
        dest="device",
        type=str,
        help="device",
        default='cuda:0'
    )
    
    parser.add_argument(
        '--num-workers',
        dest="num_workers",
        type=int,
        help="num_workers for dataloader",
        default=4
    )
    
    parser.add_argument(
        '--batch-size',
        dest="batch_size",
        type=int,
        help="batch size for dataloader",
        default=4
    )
    
    parser.add_argument(
        '--epochs',
        dest="epochs",
        type=int,
        help="epochs",
        default=300
    )
    
    parser.add_argument(
        '--log-para',
        dest="log_para",
        type=int,
        help="scaler",
        default=1000
    )
    
    parser.add_argument(
        '--lr',
        dest="lr",
        type=float,
        help="lr",
        default=1e-4
    )
    
    parser.add_argument(
        '--checkpoint-path',
        dest="checkpoint_path",
        type=str,
        help="model weights location",
    )   
    
    args = parser.parse_args()
    
    
    train_dataset = Crop_Dataset(path=args.path,
                             image_folder=args.train_image_folder,dmap_folder=args.train_dmap_folder,
                             transforms=[get_train_transforms(),get_train_image_only_transforms()],
                             crop_size=args.crop_size, ext=args.ext, dmap_ext = args.dmap_ext
                            )

    valid_dataset = Crop_Dataset(path=args.path,
                             image_folder=args.valid_image_folder,dmap_folder=args.valid_dmap_folder,
                             transforms=[get_valid_transforms()],
                             method='valid',ext=args.ext, dmap_ext = args.dmap_ext, # "Dmap.pkl"
                             crop_size = args.crop_size
                            )
    
    
    
    class TrainGlobalConfig:
        num_workers = args.num_workers
        batch_size = args.batch_size
        n_epochs = args.epochs 
        crop_size = 224
        LOG_PARA = args.log_para
        pad_mode = True
        lr = 0.0001

        folder = args.folder
        base_dir = args.path

        # -------------------
        verbose = True
        verbose_step = 1
        # -------------------

        # --------------------
        step_scheduler = True  # do scheduler.step after optimizer.step
        validation_scheduler = False
        checkpoint_path = args.checkpoint_path
        SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
        scheduler_params = dict(
            max_lr=args.lr,
            epochs=n_epochs,
            steps_per_epoch=int(len(train_dataset) / batch_size),
            pct_start=0.2,
            anneal_strategy='cos', 
            final_div_factor=10**5
        )
        
    backbone = nn.Sequential(*list(timm.create_model('swin_base_patch4_window7_224', 
                                                     pretrained=True, num_classes=0).children())[:-2])
    act_cls = nn.ReLU
    idx = [1,3,21,23]

    net = CounTR(backbone, act_cls=act_cls,
             imsize=224, layer_idx=idx).to(args.device)
    


    def run_training():
        device = torch.device(args.device)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TrainGlobalConfig.batch_size,
            sampler=RandomSampler(train_dataset),
            pin_memory=False,
            drop_last=True,
            num_workers=TrainGlobalConfig.num_workers,
            collate_fn=collate_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            batch_size=1,
            num_workers=TrainGlobalConfig.num_workers,
            shuffle=False,
            sampler=SequentialSampler(valid_dataset),
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
        fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
        fitter.fit(train_loader, val_loader)
        
    run_training()
    
    