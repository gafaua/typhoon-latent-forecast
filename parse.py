# TODO add argument parser

import configparser
from argparse import ArgumentParser
from time import time


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-c", "--config_file", type=str)
    parser.add_argument("--experiment", type=str)

    parser.add_argument("--backbone", type=str)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--out_dim", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--preprocessed_path", type=str, default=None)

    parser.add_argument("--ts_model", type=str, default=None)

    # LSTM Time series
    parser.add_argument("--labels", default="")
    parser.add_argument("--labels_input", default="")
    parser.add_argument("--labels_output", default="")
    parser.add_argument("--num_inputs", type=int, default=12)
    parser.add_argument("--num_outputs", type=int, default=1)
    parser.add_argument("--interval", type=int, default=3)
    parser.add_argument("--use_date", type=bool, default=False)
    parser.add_argument("--pred_diff", type=bool, default=False)

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--es_patience", type=int, default=-1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=str(time()))
    parser.add_argument("--checkpoint", type=str, default=None)

    # MoCo
    parser.add_argument("--queue_size", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=0.07)

    # MoCo Seq Scheduler
    parser.add_argument("--ws_range", default="")
    parser.add_argument("--ws_warmup", type=int)
    parser.add_argument("--ws_last", type=int)

    args = parser.parse_args()

    if args.config_file:
        config = configparser.ConfigParser()
        config.read(args.config_file)
        defaults = {}
        sections = ["Typhoon"]
        for s in sections:
            if config.has_section(s):
                defaults.update(dict(config.items(s)))
        parser.set_defaults(**defaults)
        args = parser.parse_args() # Overwrite arguments

    # Post processing on arguments
    args.labels = args.labels.split(",")
    args.labels_input = [int(x) for x in args.labels_input.split(",")] if len(args.labels_input) > 0 else ""
    args.labels_output = [int(x) for x in args.labels_output.split(",")] if len(args.labels_output) > 0 else ""

    args.ws_range = [int(x) for x in args.ws_range.split(",")] if len(args.ws_range) > 0 else ""

    return args
