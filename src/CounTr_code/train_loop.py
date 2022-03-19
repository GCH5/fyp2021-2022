import math
import os
import torch.nn as nn
import torch
from datetime import datetime
import time
from utils import AverageMeter
from glob import glob


def MSELoss(preds, targs):
    return nn.MSELoss()(preds, targs)


class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'{self.config.base_dir}/{self.config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device


        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr)

        self.scaler = torch.cuda.amp.GradScaler()

        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params)
        self.criterion = MSELoss
        if(self.config.checkpoint_path != None):
            self.load(self.config.checkpoint_path)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, train_mae, train_mse = self.train_one_epoch(
                train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, mse_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, mae: {train_mae.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, mse: {math.sqrt(train_mse.avg):.8f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, val_mae, val_mse = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, mse_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, mae: {val_mae.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, mse: {math.sqrt(val_mse.avg):.8f}, time: {(time.time() - t):.5f}')

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(
                    f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        val_maes = AverageMeter()
        val_mses = AverageMeter()
        crop_size = self.config.crop_size

        t = time.time()
        for step, (images, density_maps, fnames) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' +
                        f'mse_loss: {summary_loss.avg:.8f}, ' +
                        f'mae: {val_maes.avg:.8f}, ' +
                        f'mse: {math.sqrt(val_mses.avg):.8f}, ' +
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                batch_size = images.shape[0]
                if self.device == torch.device('cpu'):
                    images = images.float()
                    density_maps = density_maps.float()
                else:
                    images = images.cuda().float()
                    density_maps = density_maps.cuda().float()

                if self.config.pad_mode == True:
                    bs, ch, h, w = images.shape
                    m, n = int(h/crop_size), int(w/crop_size)
                    loss, preds_cnt, dmaps_cnt = 0, 0, 0

                with torch.cuda.amp.autocast():  # native fp16
                    for i in range(m):
                        for j in range(n):
                            img_patches = images[:, :, crop_size *
                                                 i:crop_size*(i+1), crop_size*j:crop_size*(j+1)]
                            dmaps_patches = density_maps[:, :, crop_size *
                                                         i:crop_size*(i+1), crop_size*j:crop_size*(j+1)]
                            preds = self.model(img_patches)
                            loss += self.criterion(preds, dmaps_patches)
                            preds_cnt += (preds.cpu() /
                                          self.config.LOG_PARA).sum()
                            dmaps_cnt += (dmaps_patches.cpu() /
                                          self.config.LOG_PARA).sum()
                summary_loss.update(loss.detach().item(), batch_size)
                val_maes.update(abs(dmaps_cnt-preds_cnt))
                val_mses.update((preds_cnt-dmaps_cnt)*(preds_cnt-dmaps_cnt))

        return summary_loss, val_maes, val_mses

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        train_maes = AverageMeter()
        train_mses = AverageMeter()
        crop_size = self.config.crop_size

        t = time.time()
        for step, (images, density_maps, fnames) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' +
                        f'mse_loss: {summary_loss.avg:.8f}, ' +
                        f'mae: {train_maes.avg:.8f}, ' +
                        f'mse: {math.sqrt(train_mses.avg):.8f}, ' +
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            images = images.cuda().float()
            batch_size = images.shape[0]
            density_maps = density_maps.cuda().float()
            self.optimizer.zero_grad()

            if self.config.pad_mode == True:
                _, _, h, w = images.shape
                m, n = int(h/crop_size), int(w/crop_size)
                loss, preds_cnt, dmaps_cnt = 0, 0, 0

            with torch.cuda.amp.autocast():  # native fp16
                for i in range(m):
                    for j in range(n):
                        img_patches = images[:, :, crop_size *
                                             i:crop_size*(i+1), crop_size*j:crop_size*(j+1)]
                        dmaps_patches = density_maps[:, :, crop_size *
                                                     i:crop_size*(i+1), crop_size*j:crop_size*(j+1)]
                        preds = self.model(img_patches)
                        loss += self.criterion(preds, dmaps_patches)
                        preds_cnt += (preds.cpu()/self.config.LOG_PARA).sum()
                        dmaps_cnt += (dmaps_patches.cpu() /
                                      self.config.LOG_PARA).sum()

            self.scaler.scale(loss).backward()
            summary_loss.update(loss.detach().item(), batch_size)
            train_maes.update(abs(dmaps_cnt - preds_cnt))
            train_mses.update((preds_cnt - dmaps_cnt) * (preds_cnt - dmaps_cnt))

            self.scaler.step(self.optimizer)  # native fp16

            if self.config.step_scheduler:
                self.scheduler.step()

            self.scaler.update()  # native fp16

        return summary_loss, train_maes, train_mses

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
