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
import argparse

""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="mps")

## dataset
parser.add_argument("--dataset", type=str, default="FEMNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=62)
parser.add_argument("--num_pixel", type=int, default=28)
parser.add_argument("--model", type=str, default="CNN")

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
parser.add_argument("--pruning_threshold", type=float, default=0.001)

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"


dir = os.getcwd() + "/datasets/RawData/%s" % (args.dataset)


def get_data(comm: MPI.Comm):
    # test data for a server

    test_data_raw = {}
    test_data_input = []
    test_data_label = []
    for idx in range(36):
        with open("%s/test/all_data_%s_niid_05_test_9.json" % (dir, idx)) as f:
            test_data_raw[idx] = json.load(f)
        for client in test_data_raw[idx]["users"]:
            for data_input in test_data_raw[idx]["user_data"][client]["x"]:
                data_input = np.asarray(data_input)
                data_input.resize(args.num_pixel, args.num_pixel)
                # Repeating 1 channel data to use pretrained weight that based on 3 channels data
                if args.num_channel == 1 and args.pretrained > 0:
                    test_data_input.append([data_input, data_input, data_input])
                else:
                    test_data_input.append([data_input])

            for data_label in test_data_raw[idx]["user_data"][client]["y"]:
                test_data_label.append(data_label)
    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = {}
    train_datasets = []
    num_clients = comm.Get_size()
    client_id = comm.Get_rank()
    files_per_client = 36 // num_clients
    start_file_id = client_id * files_per_client
    end_file_id = start_file_id + files_per_client

    for idx in range(start_file_id, end_file_id):
        with open("%s/train/all_data_%s_niid_05_train_9.json" % (dir, idx)) as f:
            train_data_raw[idx] = json.load(f)

        for client in train_data_raw[idx]["users"]:
            train_data_input_resize = []
            for data_input in train_data_raw[idx]["user_data"][client]["x"]:
                data_input = np.asarray(data_input)
                data_input.resize(args.num_pixel, args.num_pixel)
                # Repeating 1 channel data to use pretrained weight that based on 3 channels data
                if args.num_channel == 1 and args.pretrained > 0:
                    train_data_input_resize.append([data_input, data_input, data_input])
                else:
                    train_data_input_resize.append([data_input])

            train_datasets.append(
                Dataset(
                    torch.FloatTensor(train_data_input_resize),
                    torch.tensor(train_data_raw[idx]["user_data"][client]["y"]),
                )
            )

    return train_datasets, test_dataset


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
    cfg.compressor_lib_path = "/Users/grantwilkins/SZ3/build/tools/sz3c/libSZ3c.dylib"
    cfg.compressor_error_bound = args.error_bound
    cfg.compressor_error_mode = args.compressor_error_mode
    cfg.pruning = args.pruning
    cfg.pruning_threshold = args.pruning_threshold
    start_time = time.time()

    train_datasets, test_dataset = get_data(comm)

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
    if args.federation_type == "Federated":
        cfg.fed.args.optim = args.client_optimizer
        cfg.fed.args.optim_args.lr = args.client_lr
        cfg.fed.servername = args.server
        cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## outputs
    cfg.use_tensorboard = True

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
