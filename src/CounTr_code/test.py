#!/usr/bin/env python
from data.transforms import *
from data.crowd_dataset import Crop_Dataset
import timm
import matplotlib.pyplot as plt
from model.counTR import CounTR
import torch
import torch.nn as nn
from torch.utils.data.sampler import SequentialSampler
from train_loop import *
from utils import collate_fn
import argparse
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(filename="validation.log", level=logging.INFO)

import warnings
warnings.filterwarnings("ignore")
torch.hub.set_dir('/data1/xluap/torch/hub')

def MSELoss(preds, targs):
    return nn.MSELoss()(preds, targs)


class Validator:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.criterion = MSELoss
        
    def validate(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        val_maes = AverageMeter()
        val_mses = AverageMeter()
        crop_size = self.config.crop_size

        t = time.time()
        for step, (images, density_maps, fnames) in enumerate(tqdm(val_loader)):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    logging.info(
                        f'Val Step {step + 1}/{len(val_loader)}, ' +
                        f'mse_loss: {summary_loss.avg:.8f}, ' +
                        f'mae: {val_maes.avg:.8f}, ' +
                        f'mse: {math.sqrt(val_mses.avg):.8f}, ' +
                        f'time: {(time.time() - t):.5f}'
                    )
            continue_flag = False
            with torch.no_grad():
                batch_size = images.shape[0]
                if self.device == torch.device('cpu'):
                    images = images.float()
                    density_maps = density_maps.float()
                else:
                    images = images.cuda().float()
                    density_maps = density_maps.cuda().float()
                if math.isnan(density_maps.sum()) and continue_flag == False:
                    print(f"density_maps nan on {fnames[0]}")
                    continue_flag = True
                    # np.save('nan_dmaps.npy', density_maps.cpu().numpy())
                    
                if self.config.pad_mode == True:
                    bs, ch, h, w = images.shape
                    m, n = int(h/crop_size), int(w/crop_size)
                    loss, preds_cnt, dmaps_cnt = 0, 0, 0
                preds_whole = torch.zeros([h, w])
                with torch.cuda.amp.autocast():  # native fp16
                    for i in range(m):
                        for j in range(n):
                            img_patches = images[:, :, crop_size *
                                                    i:crop_size*(i+1), crop_size*j:crop_size*(j+1)]
                            dmaps_patches = density_maps[:, :, crop_size *
                                                            i:crop_size*(i+1), crop_size*j:crop_size*(j+1)]
                            if math.isnan(dmaps_patches.sum()) and continue_flag == False:
                                print(f"dmaps_patches nan on {fnames[0]}, {i}, {j}")
                                continue_flag = True
                                # np.save('nan_dmaps.npy', dmaps_patches.cpu().numpy())
                            preds = self.model(img_patches)
                            preds_whole[crop_size * i:crop_size*(i+1), crop_size*j:crop_size*(j+1)] = preds.cpu() / self.config.LOG_PARA
                            loss += self.criterion(preds, dmaps_patches)
                            preds_cnt += (preds.cpu() /
                                            self.config.LOG_PARA).sum()
                            dmaps_cnt += (dmaps_patches.cpu() /
                                            self.config.LOG_PARA).sum()
                            if math.isnan(preds_cnt) and continue_flag == False:
                                print(f"preds nan on {fnames[0]}")
                                continue_flag = True
                            if math.isnan(dmaps_cnt) and continue_flag == False:
                                print(f"dmaps_cnt nan on {fnames[0]}")
                                continue_flag = True
                if continue_flag:
                    continue
                summary_loss.update(loss.detach().item(), batch_size)
                val_maes.update(abs(dmaps_cnt-preds_cnt))
                val_mses.update((preds_cnt-dmaps_cnt)*(preds_cnt-dmaps_cnt))
                logging.info(f'fname:{os.path.basename(fnames[0])}, Pred: {preds_cnt.item()}, GT: {dmaps_cnt.item()}, mae:{val_maes.avg:.8f}, mse:{math.sqrt(val_mses.avg):.8f}')

                X,Y = np.meshgrid(np.linspace(1,w,w),np.linspace(h,1,h)) 
                plt.clf()
                plt.pcolormesh(X,Y,preds_whole.numpy())
                plt.title(f'GT: {dmaps_cnt.item()} Pred: {preds_cnt.item()}', fontweight ="bold") 
                plt.savefig(os.path.splitext(fnames[0])[0]+'_Pred.png')
                plt.clf()
                plt.pcolormesh(X,Y,density_maps.cpu()[0][0].numpy())
                plt.title('GT', fontweight ="bold") 
                plt.savefig(os.path.splitext(fnames[0])[0]+'_GT.png')

        return summary_loss, val_maes, val_mses
    def load(self, path):
        checkpoint = torch.load(path)
        logging.info('load model from {}'.format(path))
        self.model.load_state_dict(checkpoint['model_state_dict'])

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
        '--log-para',
        dest="log_para",
        type=int,
        help="scaler",
        default=1000
    )    
    
    parser.add_argument(
        '--checkpoint-path',
        dest="checkpoint_path",
        type=str,
        help="model weights location",
    )

    args = parser.parse_args()

    class TrainGlobalConfig:
        crop_size = 224
        LOG_PARA = args.log_para
        pad_mode = True
        num_workers = 4
        # -------------------
        verbose = True
        verbose_step = 1
        # -------------------
        checkpoint_path = args.checkpoint_path

        # --------------------        


    valid_dataset = Crop_Dataset(path=args.path,
                                 image_folder=args.valid_image_folder, dmap_folder=args.valid_dmap_folder,
                                 transforms=[get_valid_transforms()],
                                 method='valid', ext=args.ext, dmap_ext=args.dmap_ext,  # "Dmap.pkl"
                                 crop_size=args.crop_size
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

    backbone = nn.Sequential(*list(timm.create_model('swin_base_patch4_window7_224', 
                                                     pretrained=True, num_classes=0).children())[:-2])
    act_cls = nn.ReLU
    idx = [1,3,21,23]

    net = CounTR(backbone, act_cls=act_cls,
             imsize=224, layer_idx=idx).to(args.device)
    device = torch.device(args.device)
    validator = Validator(net, device, config=TrainGlobalConfig)
    validator.load(args.checkpoint_path)
    validator.validate(val_loader)
