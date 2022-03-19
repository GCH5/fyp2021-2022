#!/usr/bin/env python
from data.transforms import *
from data.crowd_dataset import Crop_Dataset_Infer, Crop_Dataset_Video
import timm
from model.counTR import CounTR
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SequentialSampler
from train_loop import *
import argparse
import logging
from pathlib import Path
import cv2

logging.basicConfig(filename="basic_log.log", level=logging.DEBUG)

import warnings

warnings.filterwarnings("ignore")
torch.hub.set_dir("/data1/xluap/torch/hub")


def MSELoss(preds, targs):
    return nn.MSELoss()(preds, targs)


class Validator:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

    def validate(self, val_loader):
        logging.debug("start validation")
        self.model.eval()
        crop_size = self.config.crop_size

        t = time.time()
        for step, images in enumerate(val_loader):
            print(f"Val Step {step+1}/{len(val_loader)}")
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    logging.info(
                        f"Val Step {step}/{len(val_loader)}, "
                        + f"time: {(time.time() - t):.5f}"
                    )
            preds_whole = torch.zeros([images.shape[2], images.shape[3]])
            with torch.no_grad():
                images = images.cuda().float()
                if self.config.pad_mode == True:
                    _, _, h, w = images.shape
                    m, n = int(h / crop_size), int(w / crop_size)
                    preds_cnt = 0
                with torch.cuda.amp.autocast():  # native fp16
                    for i in range(m):
                        for j in range(n):
                            img_patches = images[
                                :,
                                :,
                                crop_size * i : crop_size * (i + 1),
                                crop_size * j : crop_size * (j + 1),
                            ]
                            preds = self.model(img_patches)
                            preds_whole[
                                crop_size * i : crop_size * (i + 1),
                                crop_size * j : crop_size * (j + 1),
                            ] = (
                                preds.cpu() / self.config.LOG_PARA
                            )
                            preds_cnt += (preds.cpu() / self.config.LOG_PARA).sum()
                logging.info(f"Pred: {preds_cnt.item()}")

    def infer(
        self,
        infer_loader,
        output_dir="queue_analysis_out",
        output_filename="queue_analysis_out",
        fps=30,
    ):
        logging.debug("start inference")
        self.model.eval()
        crop_size = self.config.crop_size
        dir_path = Path(output_dir)
        file_path = Path(f"{output_filename}.json")
        dir_path.mkdir(exist_ok=True)
        p = Path(output_dir) / file_path
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            str(dir_path / Path(f"{output_filename}.mp4")), fourcc, fps, size
        )

        with p.open("w") as f:
            f.write("[\n")  # start of the json file
            for images, _ in infer_loader:
                preds_whole = torch.zeros([images.shape[2], images.shape[3]])
                with torch.no_grad():
                    images = images.cuda().float()
                    if self.config.pad_mode == True:
                        _, _, h, w = images.shape
                        m, n = int(h / crop_size), int(w / crop_size)
                        preds_cnt = 0
                    with torch.cuda.amp.autocast():  # native fp16
                        for i in range(m):
                            for j in range(n):
                                img_patches = images[
                                    :,
                                    :,
                                    crop_size * i : crop_size * (i + 1),
                                    crop_size * j : crop_size * (j + 1),
                                ]
                                preds = self.model(img_patches)
                                preds_whole[
                                    crop_size * i : crop_size * (i + 1),
                                    crop_size * j : crop_size * (j + 1),
                                ] = (
                                    preds.cpu() / self.config.LOG_PARA
                                )
                                preds_cnt += (preds.cpu() / self.config.LOG_PARA).sum()
                    f.write(f"{preds_cnt.item()}\n")
                X, Y = np.meshgrid(np.linspace(1, w, w), np.linspace(h, 1, h))
                fig = plt.figure()
                fig.add_subplot(111)
                fig.pcolormesh(X, Y, preds_whole.numpy())
                fig.title(
                    f"Pred: {preds_cnt.item()}",
                    fontweight="bold",
                )
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                video_writer.write(data)
            f.write("]")  # end of the json file
        video_writer.release()

    def load(self, path):
        logging.info("load model from {}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="counTR, transformer based crowd counting model"
    )

    parser.add_argument(
        "--valid-image-folder",
        dest="valid_image_folder",
        type=str,
        help="The directory name of the valid image data",
    )

    parser.add_argument(
        "--crop_size",
        dest="crop_size",
        type=int,
        help="crop_size for train image",
        default=448,
    )

    parser.add_argument(
        "--ext",
        dest="ext",
        type=str,
        help="The extension of input image",
        default=".jpg",
    )

    parser.add_argument(
        "--device", dest="device", type=str, help="device", default="cuda:0"
    )

    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="num_workers for dataloader",
        default=4,
    )

    parser.add_argument("--epochs", dest="epochs", type=int, help="epochs", default=300)

    parser.add_argument(
        "--log-para", dest="log_para", type=int, help="scaler", default=1000
    )

    parser.add_argument(
        "--checkpoint-path",
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

    valid_dataset = Crop_Dataset_Infer(
        image_folder=args.valid_image_folder,
        transforms=[get_valid_transforms()],
        ext=args.ext,  # "Dmap.pkl"
        crop_size=args.crop_size,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=True,
    )

    backbone = nn.Sequential(
        *list(
            timm.create_model(
                "swin_base_patch4_window7_224", pretrained=True, num_classes=0
            ).children()
        )[:-2]
    )
    act_cls = nn.ReLU
    idx = [1, 3, 21, 23]

    net = CounTR(backbone, act_cls=act_cls, imsize=224, layer_idx=idx).to(args.device)
    device = torch.device(args.device)
    validator = Validator(net, device, config=TrainGlobalConfig)
    validator.load(args.checkpoint_path)
    validator.validate(val_loader)


def run(
    source,
    checkpoint_path,
    device,
    output_dir,
    output_filename,
):
    valid_dataset = Crop_Dataset_Video(
        video_path=source,
        transforms=[get_valid_transforms()],
        crop_size=args.crop_size,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=True,
    )

    backbone = nn.Sequential(
        *list(
            timm.create_model(
                "swin_base_patch4_window7_224", pretrained=True, num_classes=0
            ).children()
        )[:-2]
    )
    act_cls = nn.ReLU
    idx = [1, 3, 21, 23]

    net = CounTR(backbone, act_cls=act_cls, imsize=224, layer_idx=idx).to(device)
    device = torch.device(device)
    validator = Validator(net, device, config=TrainGlobalConfig)
    validator.load(checkpoint_path)
    validator.infer(val_loader, output_dir, output_filename)
