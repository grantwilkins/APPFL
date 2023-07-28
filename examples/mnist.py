import os
import time

import numpy as np
import torch
from torch.utils.data import Subset

import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.utils import get_model

import appfl.run_serial as rs
import appfl.run_mpi as rm
from mpi4py import MPI

import argparse

""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="mps")

## dataset
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--num_pixel", type=int, default=28)
parser.add_argument("--model", type=str, default="AlexNetMNIST")
parser.add_argument("--pretrained", type=int, default=0)


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
parser.add_argument(
    "--compressor_lib_path",
    type=str,
    default="/Users/grantwilkins/SZ3/build/tools/sz3c/libSZ3c.dylib",
)
parser.add_argument("--pruning_threshold", type=float, default=0.01)


args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"


def create_dataset(data_raw):
    data_input = [x[0] for x in data_raw]
    data_label = [x[1] for x in data_raw]
    return Dataset(torch.stack(data_input), torch.tensor(data_label))


def get_data(comm: MPI.Comm):
    dir = os.getcwd() + "/datasets/RawData"
    comm_rank = comm.Get_rank()

    transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),  # MNIST normalization parameters
        ]
    )

    # Root download the data if not already available.
    if comm_rank == 0:
        # test data for a server
        test_data_raw = torchvision.datasets.MNIST(
            dir, download=True, train=False, transform=transform
        )
    comm.Barrier()
    if comm_rank > 0:
        # test data for a server
        test_data_raw = torchvision.datasets.MNIST(
            dir, download=False, train=False, transform=transform
        )

    test_dataset = create_dataset(test_data_raw)

    # training data for multiple clients
    train_data_raw = torchvision.datasets.MNIST(
        dir, download=False, train=True, transform=transform
    )

    indices = np.array_split(range(len(train_data_raw)), args.num_clients)
    train_datasets = []
    for i in range(args.num_clients):
        subset_data_raw = Subset(train_data_raw, indices[i])
        train_datasets.append(create_dataset(subset_data_raw))

    return train_datasets, test_dataset


## Run
def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    """ Configuration """
    cfg = OmegaConf.structured(Config)

    #### COMPRESSION TESTING ####
    cfg.compressed_weights_client = args.compressed_client
    cfg.compressed_weights_server = args.compressed_server
    cfg.compressor = args.compressor
    cfg.compressor_lib_path = args.compressor_lib_path
    cfg.compressor_error_bound = args.error_bound
    cfg.compressor_error_mode = args.compressor_error_mode
    cfg.pruning = args.pruning
    cfg.pruning_threshold = args.pruning_threshold
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    cfg.device = args.device
    cfg.num_clients = args.num_clients
    cfg.num_epochs = args.num_epochs

    cfg.fed = eval(args.federation_type + "()")
    if args.federation_type == "Federated":
        cfg.fed.args.optim = args.client_optimizer
        cfg.fed.args.optim_args.lr = args.client_lr
        cfg.fed.servername = args.server
        cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## dataset
    cfg.dataset = args.dataset

    ## model
    cfg.model = args.model
    ## outputs

    cfg.use_tensorboard = False

    cfg.save_model_state_dict = False

    cfg.output_dirname = "./outputs_%s_%s_%s" % (
        args.dataset,
        args.server,
        args.client_optimizer,
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
    train_datasets, test_dataset = get_data(comm)

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

    """ Running """
    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(
                cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset
            )
        else:
            rm.run_client(
                cfg,
                comm,
                model,
                loss_fn,
                args.num_clients,
                train_datasets,
                test_dataset,
            )
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)


if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 2 --mca opal_cuda_support 1 python ./mnist.py
# To run MPI:
# mpiexec -np 2 python ./mnist.py
# To run:
# python ./mnist.py
