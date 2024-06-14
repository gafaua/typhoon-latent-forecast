from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from lib.models.feature_extractors import get_feature_extractor
from lib.models.siamese_ema import SiameseEMA, infoNCELoss
from lib.trainers.base import BaseTrainer


class MocoTrainer(BaseTrainer):
    def __init__(self, train_loader, val_loader, time_window_scheduler, args) -> None:
        self.model = SiameseEMA(
            base_encoder=get_feature_extractor(args.backbone),
            out_dim=args.out_dim,
        )
        self.model = self.model.to(args.device)

        super().__init__(train_loader, val_loader, args)

        if self.use_wandb:
            wandb.init(
                project="typhoon",
                group="moco",
                name=args.run_name,
                config=args.__dict__,
            )

        self.queue_size = args.queue_size

        self.queue = torch.randn(args.out_dim, self.queue_size, device=args.device)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

        self.queue_val = torch.randn(args.out_dim, 128, device=args.device)
        self.queue_val = nn.functional.normalize(self.queue_val, dim=0)
        self.queue_ptr_val = torch.zeros(1, dtype=torch.long)

        self.T = args.temperature
        self.time_window_scheduler = time_window_scheduler


    def _train_batch_loss(self, batch, train=True):
        x1, x2 = batch#self.augment(imgs.to(self.device))
        q, k = self.model(x1.to(self.device), x2.to(self.device))

        queue = self.queue if train else self.queue_val
        queue_ptr = self.queue_ptr if train else self.queue_ptr_val

        loss = infoNCELoss(q, k, queue, self.T)

        self._dequeue_and_enqueue(k, queue, queue_ptr)

        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        queue_size = queue.shape[1]
        assert queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % queue_size  # move pointer

        queue_ptr[0] = ptr


    def _run_train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Training {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=deque(maxlen=self.log_interval))

        for i, batch in enumerate(pbar):
            self.opt.zero_grad()
            loss = self._train_batch_loss(batch)

            loss.backward()
            self.opt.step()

            self.step += 1
            losses["loss"].append(loss.item())
            running_avg = np.mean(losses["loss"])
            pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))

            if self.use_wandb and self.step % self.log_interval == 0:
                wandb.log(data={"train_loss": running_avg}, step=self.step)

        self.lr_scheduler.step()
        self.time_window_scheduler.step()

    def _run_val_epoch(self):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"Eval {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=list())

        with torch.no_grad():
            for i, batch in enumerate(pbar):
                loss = self._train_batch_loss(batch, train=False)

                losses["loss"].append(loss.item())
                running_avg = np.mean(losses["loss"])
                pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))


        if self.use_wandb:
            wandb.log(data={"val_loss": np.mean(losses["loss"])}, step=self.step)
