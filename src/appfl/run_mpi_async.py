import copy
import time
import logging
import torch.nn as nn
from .misc import *
from mpi4py import MPI
from .algorithm import *
from torch.optim import *
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from .comm.mpi.mpi_communicator import MpiCommunicator


def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "appfl",
):
    """
    run_server:
        Run PPFL server that updates the global model parameters in an asynchronous way
    args:
        cfg - the configuration for the FL experiment
        comm - MPI communicator
        model - neural network model to train
        loss_fn - loss function
        num_clients - the number of clients used in PPFL simulation
        test_dataset - optional testing data. If given, validation will run based on this data
        dataset_name - optional dataset name
        metric - evaluation metric function
    """
    device = "cpu"
    communicator = MpiCommunicator(comm)

    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)
    cfg.logginginfo.comm_size = comm.Get_size()
    cfg.logginginfo.DataSet_name = dataset_name

    "Using tensorboard to visualize the test loss."
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

    "Collect the number of data from each client and compute the weights for each client"
    num_data = communicator.gather(0, dest=0)
    total_num_data = 0
    for num in num_data:
        total_num_data += num
    weights = [num / total_num_data for num in num_data]

    "Asynchronous federated learning server"
    server = eval(cfg.fed.servername)(
        weights, copy.deepcopy(model), loss_fn, num_clients, device, **cfg.fed.args
    )

    start_time = time.time()
    iter = 0
    client_model_step = {i: 0 for i in range(0, num_clients)}
    client_start_time = {i: start_time for i in range(0, num_clients)}

    server.model.to("cpu")
    global_model = server.model.state_dict()

    "First broadcast the global model "
    communicator.broadcast_global_model(global_model)

    "Main server training loop"
    test_loss, test_accuracy, best_accuracy = 0.0, 0.0, 0.0
    while True:
        # Wait for response from any one client
        client_idx, local_model_size = MPI.Request.waitany(recv_reqs)

        if client_idx != MPI.UNDEFINED:
            # Record time
            local_start_time = client_local_time[client_idx]
            local_update_time = time.time() - client_local_time[client_idx]
            global_update_start = time.time()

            # Increment the global step
            global_step += 1
            logger.info(
                f"[Server Log] [Step #{global_step:3}] Server gets model size from client #{client_idx}"
            )

            # Allocate a buffer to receive the model byte stream
            local_model_bytes = np.empty(local_model_size, dtype=np.byte)

            # Receive the model byte stream
            comm.Recv(
                local_model_bytes, source=client_idx + 1, tag=client_idx + 1 + comm_size
            )
            logger.info(
                f"[Server Log] [Step #{global_step:3}] Server gets model from client #{client_idx}"
            )

            # Load the model byte to state dict
            local_model_buffer = io.BytesIO(local_model_bytes.tobytes())
            local_model_dict = torch.load(local_model_buffer)

            # Perform global update
            logger.info(
                f"[Server Log] [Step #{global_step:3}] Server updates global model based on the model from client #{client_idx}"
            )
            server.update(local_model_dict, client_model_step[client_idx], client_idx)
            global_update_time = time.time() - global_update_start

            # Remove the completed request from list
            recv_reqs.pop(client_idx)
            if global_step < cfg.num_epochs:
                # Convert the updated model to bytes
                global_model = server.model.state_dict()
                gloabl_model_buffer = io.BytesIO()
                torch.save(global_model, gloabl_model_buffer)
                global_model_bytes = gloabl_model_buffer.getvalue()

                # Send (buffer size, finish flag) - INFO - to the client in a blocking way
                comm.send(
                    (len(global_model_bytes), False),
                    dest=client_idx + 1,
                    tag=client_idx + 1,
                )

                # Send the buffered model - MODEL - to the client in a blocking way
                comm.Send(
                    np.frombuffer(global_model_bytes, dtype=np.byte),
                    dest=client_idx + 1,
                    tag=client_idx + 1 + comm_size,
                )

                # Add new receiving request to the list
                recv_reqs.insert(
                    client_idx, comm.irecv(source=client_idx + 1, tag=client_idx + 1)
                )

                # Update the model step for the client
                client_model_step[client_idx] = server.global_step

                # Update the local training time of the client
                client_local_time[client_idx] = time.time()

            # Do server validation
            validation_start = time.time()
            if cfg.validation == True:
                test_loss, test_accuracy = validation(server, test_dataloader)
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                if cfg.use_tensorboard:
                    # Add them to tensorboard
                    writer.add_scalar(
                        "server_test_accuracy", test_accuracy, global_step
                    )
                    writer.add_scalar("server_test_loss", test_loss, global_step)
            cfg["logginginfo"]["Validation_time"] = time.time() - validation_start
            cfg["logginginfo"]["PerIter_time"] = time.time() - local_start_time
            cfg["logginginfo"]["Elapsed_time"] = time.time() - start_time
            cfg["logginginfo"]["test_loss"] = test_loss
            cfg["logginginfo"]["test_accuracy"] = test_accuracy
            cfg["logginginfo"]["BestAccuracy"] = best_accuracy
            cfg["logginginfo"]["LocalUpdate_time"] = local_update_time
            cfg["logginginfo"]["GlobalUpdate_time"] = global_update_time
            logger.info(f"[Server Log] [Step #{global_step:3}] Iteration Logs:")
            if global_step != 1:
                logger.info(server.log_title())
            server.logging_iteration(cfg, logger, global_step - 1)

        "Break after max updates"
        if iter == cfg.num_epochs:
            break

    "Notify the clients about the end of the learning"
    communicator.cleanup()
    for i in range(num_clients):
        communicator.send_single_global_model(None, {"done": True}, i)

    "Log the summary of the FL experiments"
    server.logging_summary(cfg, logger)


def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
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
    client_idx = comm.Get_rank() - 1
    communicator = MpiCommunicator(comm)

    """ log for clients"""
    output_filename = cfg.output_filename + "_client_%s" % (client_idx - 1)
    outfile = client_log(cfg.output_dirname, output_filename)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    logger.addHandler(c_handler)

    num_data = len(train_data[client_idx - 1])
    communicator.gather(num_data, dest=0)

    batchsize = cfg.train_data_batch_size
    if cfg.batch_training == False:
        batchsize = len(train_data[client_idx - 1])

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

    client = eval(cfg.fed.clientname)(
        client_idx,
        None,  # TODO: Now I set weights to None, I don't know why we need weights?
        copy.deepcopy(model),
        loss_fn,
        DataLoader(
            train_data[client_idx],
            num_workers=cfg.num_workers,
            batch_size=batchsize,
            shuffle=True,
            pin_memory=True,
        ),
        cfg,
        outfile,
        test_dataloader,
        **cfg.fed.args,
    )

    while True:
        model = communicator.recv_global_model(source=0)
        if isinstance(model, tuple):
            model, done = model[0], model[1]["done"]
        else:
            done = False
        if done:
            break
        client.model.load_state_dict(model)
        client.update()
        # Compute gradient if the algorithm is gradient-based
        if cfg.fed.args.gradient_based:
            list_named_parameters = []
            for name, _ in client.model.named_parameters():
                list_named_parameters.append(name)
            local_model = {}
            for name in model:
                if name in list_named_parameters:
                    local_model[name] = model[name] - client.primal_state[name]
                else:
                    local_model[name] = client.primal_state[name]
        else:
            local_model = copy.deepcopy(client.primal_state)
        communicator.send_local_model(local_model, dest=0)
    outfile.close()
