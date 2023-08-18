import os
import time

import json
import numpy as np
import torch

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.cnn import *
import appfl.run_serial as rs
import appfl.run_mpi as rm
from mpi4py import MPI
from models.utils import get_model
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda
from torchvision.datasets import FashionMNIST
import argparse

""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="mps")

## dataset
parser.add_argument("--dataset", type=str, default="FMNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=62)
parser.add_argument("--num_pixel", type=int, default=28)
parser.add_argument("--model", type=str, default="AlexNetMNIST")

## algorithm
parser.add_argument(
    "--federation_type", type=str, default="Federated"
)  ## Federated, ICEADMM, IIADMM
## clients
parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=50)

parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)

parser.add_argument("--pretrained", type=int, default=0)

## compression
parser.add_argument("--error_bound", type=float, required=False, default=0.1)
# parser.add_argument("--compressed_client", type=bool, required=False, default=False)
parser.add_argument(
    "--compressed_client",
    action=argparse.BooleanOptionalAction,
    required=False,
    default=False,
)
parser.add_argument(
    "--compressed_server",
    action=argparse.BooleanOptionalAction,
    required=False,
    default=False,
)
parser.add_argument("--compressor", type=str, required=False, default="SZ3")
parser.add_argument("--compressor_error_mode", type=str, required=False, default="REL")
parser.add_argument(
    "--pruning",
    action=argparse.BooleanOptionalAction,
    required=False,
    default=False,
)
parser.add_argument("--param_cutoff", type=float, default=1024)
parser.add_argument("--pruning_threshold", type=float, default=0.5)

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"


dir = os.getcwd() + "/datasets/RawData/%s" % (args.dataset)


def get_data():
    dir = os.getcwd() + "/datasets/RawData"

    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
        ]
    )

    # Load test data
    test_dataset = FashionMNIST(dir, download=True, train=False, transform=transform)

    # Load train data
    train_dataset = FashionMNIST(dir, download=True, train=True, transform=transform)

    # Split train data for multiple clients
    train_dataset_splits = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) // args.num_clients] * args.num_clients
    )
    return train_dataset_splits, test_dataset


def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # read default configuration
    cfg = OmegaConf.structured(Config)

    ## Reproducibility
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    cfg.compressed_weights_client = args.compressed_client
    cfg.compressed_weights_server = args.compressed_server
    cfg.compressor = args.compressor
    if args.compressor == "SZ2":
        cfg.compressor_lib_path = "/Users/grantwilkins/SZ/build/sz/libSZ.dylib"
    elif args.compressor == "SZx":
        cfg.compressor_lib_path = "/Users/grantwilkins/SZx-main/build/lib/libSZx.dylib"
    else:
        cfg.compressor_lib_path = (
            "/Users/grantwilkins/SZ3/build/tools/sz3c/libSZ3c.dylib"
        )
    cfg.compressor_error_bound = args.error_bound
    cfg.compressor_error_mode = args.compressor_error_mode
    cfg.pruning = args.pruning
    cfg.pruning_threshold = args.pruning_threshold
    cfg.param_cutoff = int(args.param_cutoff)
    start_time = time.time()

    train_datasets, test_dataset = get_data()

    cfg.dataset = args.dataset
    cfg.model = args.model

    if cfg.data_sanity == True:
        data_sanity_check(
            train_datasets, test_dataset, args.num_channel, args.num_pixel
        )

    model = get_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    ## settings
    cfg.device = args.device
    cfg.num_clients = args.num_clients
    cfg.num_epochs = args.num_epochs

    cfg.fed = eval(args.federation_type + "()")
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.servername = args.server
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## outputs
    cfg.use_tensorboard = False

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(
                cfg=cfg,
                comm=comm,
                model=model,
                loss_fn=loss_fn,
                num_clients=args.num_clients,
                test_dataset=test_dataset,
                dataset_name=args.dataset,
            )
        else:
            rm.run_client(
                cfg=cfg,
                comm=comm,
                model=model,
                loss_fn=loss_fn,
                num_clients=args.num_clients,
                train_data=train_datasets,
                test_data=test_dataset,
            )
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)


if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./femnist.py
# To run MPI:
# mpiexec -np 5 python ./femnist.py
# To run:
# python ./femnist.py
# To run with resnet pretrained weight:
# python ./femnist.py --model resnet18 --pretrained 1
