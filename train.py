# TODO add main training pipeline

import random
from pprint import pprint

import numpy as np
import torch

from lib.trainers.moco import MocoTrainer
from lib.trainers.time_series import TimeSeriesTrainer
from lib.utils.dataloaders import get_dataloaders
from lib.utils.utils import SimpleScheduler
from parse import parse_args


def train():
    args = parse_args()
    pprint(args.__dict__)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    loader_kwargs = dict()
    if args.experiment == "moco_sequence":
        loader_kwargs["time_window_scheduler"] = SimpleScheduler(args.ws_range,
                                                                args.ws_warmup,
                                                                args.ws_last,
                                                                verbose=True)

    train_loader, val_loader, _ = get_dataloaders(args, **loader_kwargs)

    print("\nDATALOADERS LOADED, STARING TRAINING\n")

    if "ts" in args.experiment:
        assert args.ts_model is not None
        TimeSeriesTrainer(train_loader,
                        val_loader,
                        args).train()
    elif "moco" in args.experiment:
        MocoTrainer(train_loader,
                    val_loader,
                    loader_kwargs["time_window_scheduler"],
                    args).train()

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    train()
