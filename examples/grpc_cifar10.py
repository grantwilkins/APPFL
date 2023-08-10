import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os

from models.utils import get_model
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.cnn import *
import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client
from mpi4py import MPI
import argparse


""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="mps")

## dataset and model
parser.add_argument("--dataset", type=str, default="CIFAR10")
parser.add_argument("--num_channel", type=int, default=3)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--num_pixel", type=int, default=32)
parser.add_argument("--model", type=str, default="AlexNetCIFAR")
parser.add_argument("--pretrained", type=int, default=0)
parser.add_argument("--train_data_batch_size", type=int, default=128)
parser.add_argument("--test_data_batch_size", type=int, default=128)

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
parser.add_argument("--num_epochs", type=int, default=20)

parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)

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
parser.add_argument("--pruning_threshold", type=float, default=0.01)

args = parser.parse_args()

args.save_model_state_dict = True


def get_data():
    dir = os.getcwd() + "/datasets/RawData"

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )

    # Define common transform for train and test data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        dir, train=False, download=True, transform=transform
    )

    train_dataset = torchvision.datasets.CIFAR10(
        dir, train=True, download=True, transform=transform
    )

    train_length = len(train_dataset)
    per_client_length = train_length // args.num_clients
    remainder = train_length % args.num_clients

    # The first 'remainder' splits have length 'per_client_length+1', the rest have length 'per_client_length'
    lengths = [per_client_length + 1] * remainder + [per_client_length] * (
        args.num_clients - remainder
    )

    train_dataset_splits = torch.utils.data.random_split(train_dataset, lengths)

    return train_dataset_splits, test_dataset


def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    ## Reproducibility
    set_seed(1)

    """ Configuration """
    cfg = OmegaConf.structured(Config)

    cfg.device = args.device
    cfg.save_model_state_dict = args.save_model_state_dict

    cfg.compressed_weights_client = args.compressed_client
    cfg.compressed_weights_server = args.compressed_server
    cfg.compressor = args.compressor
    cfg.compressor_lib_path = "/Users/grantwilkins/SZ3/build/tools/sz3c/libSZ3c.dylib"
    cfg.compressor_error_bound = args.error_bound
    cfg.compressor_error_mode = args.compressor_error_mode
    cfg.pruning = args.pruning
    cfg.pruning_threshold = args.pruning_threshold

    ## dataset
    cfg.train_data_batch_size = args.train_data_batch_size
    cfg.test_data_batch_size = args.test_data_batch_size
    cfg.train_data_shuffle = True

    ## dataset
    cfg.dataset = args.dataset

    ## model
    cfg.model = args.model

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs
    cfg.output_dirname = "./outputs_%s_%s_%s_%s_%s_%s" % (
        args.dataset,
        args.model,
        args.server,
        args.client_optimizer,
        args.num_local_epochs,
        args.client_lr,
    )
    if args.server_lr != None:
        cfg.fed.args.server_learning_rate = args.server_lr
        cfg.output_dirname += "_ServerLR_%s" % (args.server_lr)

    if args.adapt_param != None:
        cfg.fed.args.server_adapt_param = args.adapt_param
        cfg.output_dirname += "_AdaptParam_%s" % (args.adapt_param)

    if args.mparam_1 != None:
        cfg.fed.args.server_momentum_param_1 = args.mparam_1
        cfg.output_dirname += "_MParam1_%s" % (args.mparam_1)

    if args.mparam_2 != None:
        cfg.fed.args.server_momentum_param_2 = args.mparam_2
        cfg.output_dirname += "_MParam2_%s" % (args.mparam_2)

    cfg.output_filename = "result"

    start_time = time.time()

    """ User-defined model """
    model = get_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()

    ## loading models
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname = "./save_models"
        cfg.load_model_filename = "Model"
        model = load_model(cfg)

    """ User-defined data """
    train_datasets, test_dataset = get_data()

    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(
            train_datasets, test_dataset, args.num_channel, args.num_pixel
        )

    print(
        "-------Loading_Time=",
        time.time() - start_time,
    )

    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname = "./save_models"
        cfg.save_model_filename = "Model"

    if comm_size > 1:
        # Try to launch both a server and clients.
        if comm_rank == 0:
            grpc_server.run_server(
                cfg=cfg,
                model=model,
                loss_fn=loss_fn,
                num_clients=cfg.num_clients,
                test_data=test_dataset,
            )
        else:
            grpc_client.run_client(
                cfg=cfg,
                cid=comm_rank - 1,
                model=model,
                loss_fn=loss_fn,
                train_data=train_datasets[comm_rank - 1],
                gpu_id=comm_rank,
                test_data=test_dataset,
            )
        print("------DONE------", comm_rank)
    else:
        # Just launch a server.
        grpc_server.run_server(
            cfg=cfg,
            model=model,
            loss_fn=loss_fn,
            num_clients=cfg.num_clients,
            test_data=test_dataset,
        )


if __name__ == "__main__":
    main()
