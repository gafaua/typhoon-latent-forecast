from abc import abstractmethod
from os import makedirs

import torch


class BaseTrainer:
    def __init__(self,
                 train_loader,
                 val_loader,
                 args) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = args.device

        self.batch_size = args.batch_size
        self.lr = args.lr

        self.num_epochs = args.num_epochs
        self.num_workers = args.num_workers

        self.use_wandb = args.use_wandb
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval

        self.save_dir = f"{args.save_dir}/{args.experiment}/{args.run_name}"
        makedirs(self.save_dir, exist_ok=True)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs)

        if args.checkpoint is not None:
            self._load_checkpoint(args.checkpoint)
        else:
            self.model_params = list(self.model.parameters())
            self.name = args.run_name
            self.step = 0
            self.epoch = 0

        print(f"Trainer ready, model with {sum(p.numel() for p in self.model_params):,} parameters")
        print(f"Checkpoints will be saved every {self.save_interval} epochs")

        self.early_stopping = args.es_patience > 0

        if self.early_stopping:
            self.best_val_loss = float("inf")
            self.patience = args.es_patience
            self.exasperation = 0
            print(f"This run will stop after {args.es_patience} epochs with no improvement on validation")

    def train(self):
        train_epochs = range(self.epoch, self.num_epochs)
        for _ in train_epochs:
            self._run_train_epoch()

            if self.epoch % self.save_interval == 0:
                self._save_checkpoint()

            val_loss = self._run_val_epoch()
            self.epoch += 1

            if self.early_stopping:
                if val_loss < self.best_val_loss:
                    self.exasperation = 0
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")
                else:
                    self.exasperation += 1

                if self.exasperation == self.patience:
                    print("EARLY STOPPING, MAXIMUM EXASPERATION REACHED")
                    print(f"Best validation epoch: {self.epoch - self.patience}")
                    break

            print(f"LR: {self.lr_scheduler.get_last_lr()[0]:.5f}")

        self._save_checkpoint()

    @abstractmethod
    def _run_train_epoch(self):
        ...


    @abstractmethod
    def _run_val_epoch(self):
        ...


    def _save_checkpoint(self, name=None):
        model_dict = self.model.state_dict()

        data = dict(
            model_dict=model_dict,
            opt_dict=self.opt.state_dict(),
            epoch=self.epoch,
            step=self.step,
            name=self.name,
        )

        path = f"{self.save_dir}/checkpoint_{self.step if name is None else name}.pth"

        torch.save(data, path)
        print(f"Checkpoint saved in {path}")


    def _load_checkpoint(self, path):
        data = torch.load(path)
        self.model.load_state_dict(data["model_dict"])
        self.model_params = list(self.model.parameters())
        self.opt.load_state_dict(data["opt_dict"])
        self.epoch = data["epoch"]
        self.step = data["step"]
        self.name = f"{data['name']}_resumed"

        print("="*100)
        print(f"Resuming training from checkpoint {path}")
        print("="*100)
