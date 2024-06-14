from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from tqdm import tqdm

import wandb
from lib.models.lstm_predictor import LSTM
from lib.models.tcn import TCNForecasting
from lib.trainers.base import BaseTrainer


class TimeSeriesTrainer(BaseTrainer):
    def __init__(self, train_loader, val_loader, args) -> None:
        if args.ts_model == "lstm":
            self.model = LSTM(
                train_loader.dataset.dataset.get_input_size(),
                hidden_size=args.hidden_dim,
                num_layers=args.num_layers,
                output_size=train_loader.dataset.dataset.num_preds
            ).to(args.device)
        elif args.ts_model == "tcn":
            self.model = TCNForecasting(
                input_size = train_loader.dataset.dataset.get_input_size(),
                output_size = train_loader.dataset.dataset.get_output_size(),
            ).to(args.device)
        else:
            raise NotImplementedError

        super().__init__(train_loader, val_loader, args)

        if self.use_wandb:
            wandb.init(
                project="typhoon",
                group="time-series",
                name=args.run_name,
                config=args.__dict__,
            )

        self.reg_criterion = nn.MSELoss()
        self.labels = args.labels


    def _run_train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Training {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=deque(maxlen=self.log_interval))

        if self.num_latent is not None:
            losses["lat"]=deque(maxlen=self.log_interval)

        for batch in pbar:
            self.opt.zero_grad()
            inp, outs = batch[0].to(self.device), batch[1].to(self.device)
            preds = self.model(inp)
            # For transition to 6 prediction
            if "grade" in self.labels:
                outs = outs == 6
                preds = sigmoid(preds)
            loss = self.reg_criterion(preds, (outs).float().squeeze())
            losses["loss"].append(loss.item())

            loss.backward()
            self.opt.step()

            self.step += 1

            avg = {f"tr_{key}": np.mean(val) for key, val in losses.items()}
            pbar.set_postfix(dict(loss=loss.item(), **avg))

            if self.use_wandb and self.step % self.log_interval == 0:
                wandb.log(data=avg, step=self.step)

        self.lr_scheduler.step()

    def _run_val_epoch(self):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc=f"Eval {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=list())

        if self.num_latent is not None:
            losses["lat"]=deque(maxlen=self.log_interval)

        with torch.no_grad():
            for batch in pbar:
                inp, outs = batch[0].to(self.device), batch[1].to(self.device)

                preds = self.model(inp)

                if "grade" in self.labels:
                    outs = outs == 6
                    preds = sigmoid(preds)
                loss = self.reg_criterion(preds, (outs).float().squeeze())
                losses["loss"].append(loss.item())

                avg = {f"ev_{key}": np.mean(val) for key, val in losses.items()}
                pbar.set_postfix(dict(loss=loss.item(), **avg))

        if self.use_wandb:
            wandb.log(data=avg, step=self.step)

        return np.mean(losses["loss"])
