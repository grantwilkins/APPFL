from cmath import nan

from collections import OrderedDict
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader

import numpy as np

from omegaconf import DictConfig

import copy
import time
import logging

from .misc import *
from .algorithm import *

from mpi4py import MPI
from typing import Any
import copy


def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "appfl",
    metric: Any = None
):
    """Run PPFL simulation server that aggregates and updates the global parameters of model

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train
        loss_fn (nn.Module): loss function
        num_clients (int): the number of clients used in PPFL simulation
        test_data (Dataset): optional testing data. If given, validation will run based on this data.
        DataSet_name (str): optional dataset name
    """
    ## Start
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    num_client_groups = np.array_split(range(num_clients), comm_size - 1)
    compressor = None
    if cfg.compressed_weights_client == True:
        compressor = Compressor(cfg=cfg)

    # FIXME: I think it's ok for server to use cpu only.
    device = "cpu"

    """ log for a server """
    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)

    cfg["logginginfo"]["comm_size"] = comm_size
    cfg["logginginfo"]["DataSet_name"] = dataset_name

    ## Using tensorboard to visualize the test loss
    if cfg.use_tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(
            comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients)
        )

    "Run validation if test data is given or the configuration is enabled."
    if cfg.validation == True and len(test_dataset) > 0:
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False

    """
    Receive the number of data from clients
    Compute "weight[client] = data[client]/total_num_data" from a server    
    Scatter "weight information" to clients        
    """
    num_data = comm.gather(0, root=0)
    total_num_data = 0
    for rank in range(1, comm_size):
        for val in num_data[rank].values():
            total_num_data += val

    weight = []
    weights = {}
    for rank in range(comm_size):
        if rank == 0:
            weight.append(0)
        else:
            temp = {}
            for key in num_data[rank].keys():
                temp[key] = num_data[rank][key] / total_num_data
                weights[key] = temp[key]
            weight.append(temp)

    weight = comm.scatter(weight, root=0)

    # TODO: do we want to use root as a client?
    server = eval(cfg.fed.servername)(
        weights, copy.deepcopy(model), loss_fn, num_clients, device, **cfg.fed.args
    )
    comp_ratio = 0.0
    do_continue = True
    start_time = time.time()
    test_loss = 0.0
    test_accuracy = 0.0
    best_accuracy = 0.0
    for t in range(cfg.num_epochs):
        per_iter_start = time.time()
        do_continue = comm.bcast(do_continue, root=0)

        # We need to load the model on cpu, before communicating.
        # Otherwise, out-of-memeory error from GPU
        server.model.to("cpu")

        global_state = server.model.state_dict()
        server_comp_ratio = 0.0
        if cfg.compressed_weights_server == True:
            global_state = utils.flatten_model_params(model=server.model)
            original_size = global_state.shape[0]
            global_state = compressor.compress_error_control(
                ori_data=global_state,
                error_bound=np.std(global_state),
                error_mode="REL",
            )
            server_comp_ratio = original_size / (len(global_state) * 4)

        local_update_start = time.time()
        global_state = comm.bcast(global_state, root=0)

        local_states = [None for i in range(num_clients)]
        for rank in range(comm_size):
            ls = ""
            if rank == 0:
                continue
            else:
                for _, cid in enumerate(num_client_groups[rank - 1]):
                    local_states[cid] = comm.recv(source=rank, tag=cid)

        local_update_time = time.time() - local_update_start
        cfg["logginginfo"]["LocalUpdate_time"] = local_update_time
        ori_shape = utils.flatten_model_params(model=server.model).shape
        decompress_times = [0.0 for i in range(num_clients)]
        if cfg.compressed_weights_client == True:
            for local_state in local_states:
                ori_dtype = eval(cfg.flat_model_dtype)
                decompress_time_start = time.time()
                copy_arr = compressor.decompress(
                    cmp_data=local_state["primal"],
                    ori_shape=ori_shape,
                    ori_dtype=np.float32,
                )
                decompress_times.append(time.time() - decompress_time_start)
                local_state["primal"] = copy_arr
                new_state_dic = utils.unflatten_model_params(
                    model=model, flat_params=local_state["primal"]
                )
                local_state["primal"] = new_state_dic
        # print("Start Server Update")
        global_update_start = time.time()
        server.update(local_states)
        global_update_time = time.time() - global_update_start
        cfg["logginginfo"]["GlobalUpdate_time"] = global_update_time
        validation_start = time.time()
        if cfg.validation == True:
            test_loss, test_accuracy = validation(server, test_dataloader, metric)

            if cfg.use_tensorboard:
                # Add them to tensorboard
                writer.add_scalar("server_test_accuracy", test_accuracy, t)
                writer.add_scalar("server_test_loss", test_loss, t)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
        validation_time = time.time() - validation_start
        periter_time = time.time() - per_iter_start
        elapsed_time = time.time() - start_time
        cfg["logginginfo"]["Validation_time"] = validation_time
        cfg["logginginfo"]["PerIter_time"] = periter_time
        cfg["logginginfo"]["Elapsed_time"] = elapsed_time
        cfg["logginginfo"]["test_loss"] = test_loss
        cfg["logginginfo"]["test_accuracy"] = test_accuracy
        cfg["logginginfo"]["BestAccuracy"] = best_accuracy

        server.logging_iteration(cfg, logger, t)
        for cid in range(num_clients):
            stats_file = "stats_" + str(cid) + ".csv"
            with open(stats_file, "a") as f:
                f.write(
                    str(decompress_times[cid])
                    + ","
                    + str(server_comp_ratio)
                    + ","
                    + str(t)
                    + ","
                    + str(cfg.num_epochs)
                    + ","
                    + str(cfg.num_clients)
                    + ","
                    + str(validation_time)
                    + ","
                    + str(periter_time)
                    + ","
                    + str(elapsed_time)
                    + ","
                    + str(test_loss)
                    + ","
                    + str(test_accuracy)
                    + ","
                    + str(best_accuracy)
                    + ","
                    + cfg.fed.servername
                    + ","
                    + cfg.fed.args.optim
                    + "\n"
                )
        """ Saving model """
        if (t + 1) % cfg.checkpoints_interval == 0 or t + 1 == cfg.num_epochs:
            if cfg.save_model == True:
                save_model_iteration(t + 1, server.model, cfg)

        if np.isnan(test_loss) == True:
            break

    """ Summary """
    server.logging_summary(cfg, logger)

    do_continue = False
    do_continue = comm.bcast(do_continue, root=0)


def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    metric: Any = None
):
    """Run PPFL simulation clients, each of which updates its own local parameters of model

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train
        num_clients (int): the number of clients used in PPFL simulation
        train_data (Dataset): training data
        test_data (Dataset): testing data
    """
    compressor = None
    if cfg.compressed_weights_client == True:
        compressor = Compressor(cfg=cfg)
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    ## We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        device = f"cuda:{comm_rank-1}"
    else:
        device = cfg.device

    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    """ log for clients"""
    outfile = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        output_filename = cfg.output_filename + "_client_%s" % (cid)
        outfile[cid] = client_log(cfg.output_dirname, output_filename)

    """
    Send the number of data to a server
    Receive "weight_info" from a server    
        (fedavg)            "weight_info" is not needed as of now.
        (iceadmm+iiadmm)    "weight_info" is needed for constructing coefficients of the loss_function         
    """
    num_data = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        num_data[cid] = len(train_data[cid])

    comm.gather(num_data, root=0)

    weight = None
    weight = comm.scatter(weight, root=0)

    batchsize = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        batchsize[cid] = cfg.train_data_batch_size
        if cfg.batch_training == False:
            batchsize[cid] = len(train_data[cid])

    "Run validation if test data is given or the configuration is enabled."
    if cfg.validation == True and len(test_data) > 0:
        test_dataloader = DataLoader(
            test_data,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False
        test_dataloader = None

    clients = [
        eval(cfg.fed.clientname)(
            cid,
            weight[cid],
            copy.deepcopy(model),
            loss_fn,
            DataLoader(
                train_data[cid],
                num_workers=cfg.num_workers,
                batch_size=batchsize[cid],
                shuffle=cfg.train_data_shuffle,
                pin_memory=True,
            ),
            cfg,
            outfile[cid],
            test_dataloader,
            metric,
            **cfg.fed.args,
        )
        for _, cid in enumerate(num_client_groups[comm_rank - 1])
    ]

    do_continue = comm.bcast(None, root=0)
    ori_shape = flatten_model_params(model=model).shape
    while do_continue:
        """Receive "global_state" """
        global_state = comm.bcast(None, root=0)
        if cfg.compressed_weights_server == True:
            global_state = compressor.decompress(
                cmp_data=global_state, ori_shape=ori_shape, ori_dtype=np.float32
            )
            global_state = utils.unflatten_model_params(
                model=model, flat_params=global_state
            )
        """ Update "local_states" based on "global_state" """
        reqlist = []
        for client in clients:
            cid = client.id
            ## initial point for a client model
            client.model.load_state_dict(global_state)

            ## client update
            ls = client.update()
            req = comm.isend(ls, dest=0, tag=cid)
            reqlist.append(req)

        MPI.Request.Waitall(reqlist)
        do_continue = comm.bcast(None, root=0)

    for client in clients:
        client.outfile.close()
