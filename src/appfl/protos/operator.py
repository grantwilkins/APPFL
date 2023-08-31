import logging
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import numpy as np
import copy

from appfl.misc.utils import *
from appfl.algorithm import *

from .federated_learning_pb2 import Job


class FLOperator:
    def __init__(self, cfg, model, loss_fn, test_dataset, num_clients):
        self.logger1 = create_custom_logger(logging.getLogger(__name__), cfg)
        cfg["logginginfo"]["comm_size"] = 1

        self.logger = logging.getLogger(__name__)
        self.operator_id = cfg.operator.id
        self.cfg = cfg
        self.num_clients = num_clients
        self.num_epochs = cfg.num_epochs
        self.round_number = 1
        self.best_accuracy = 0.0
        self.device = "cpu"
        self.model = copy.deepcopy(model)
        self.loss_fn = loss_fn
        """ Loading Model """
        if cfg.load_model == True:
            self.model = load_model(cfg)
        self.client_training_size = OrderedDict()
        self.client_training_size_received = OrderedDict()
        self.client_weights = OrderedDict()
        self.client_states = [OrderedDict() for _ in range(self.num_clients)]
        self.client_learning_status = OrderedDict()
        self.servicer = None  # Takes care of communication via gRPC
        self.test_loss = 0.0
        self.test_accuracy = 0.0

        self.dataloader = None
        if self.cfg.validation == True and len(test_dataset) > 0:
            self.dataloader = DataLoader(
                test_dataset,
                num_workers=0,
                batch_size=cfg.test_data_batch_size,
                shuffle=cfg.test_data_shuffle,
            )
        else:
            self.cfg.validation = False

        self.fed_server: BaseServer = eval(self.cfg.fed.servername)(
            self.client_weights,
            self.model,
            self.loss_fn,
            self.num_clients,
            self.device,
            **self.cfg.fed.args,
        )

    """
    Return the tensor record of a global model requested by its name.
    """

    def get_tensor(self, name):
        return (
            np.array(self.fed_server.model.state_dict()[name])
            if name in self.fed_server.model.state_dict()
            else None
        )

    """
    Return the job status indicating the next job a client is supposed to do.
    """

    def get_job(self):
        job_todo = Job.WEIGHT
        self.logger.debug(
            f"[Round: {self.round_number: 04}] client_training_size_received: {self.client_training_size_received}"
        )
        if all(
            c in self.client_training_size_received for c in range(self.num_clients)
        ):
            job_todo = Job.TRAIN
        if self.round_number > self.num_epochs:
            job_todo = Job.QUIT
        return min(self.round_number, self.num_epochs), job_todo

    """
    Compute weights of clients based on their training data size.
    """

    def get_weight(self, client_id, training_size) -> float:
        self.client_training_size[client_id] = training_size
        self.client_training_size_received[client_id] = True
        self.logger.debug(
            f"[Round: {self.round_number: 04}] client_training_size_received: {self.client_training_size_received}"
        )

        # If we have received training size from all clients
        if all(
            c in self.client_training_size_received for c in range(self.num_clients)
        ):
            # Instantiate a fed_server if not instantiated yet.
            total_training_size = sum(
                self.client_training_size[c] for c in range(self.num_clients)
            )
            for c in range(self.num_clients):
                self.client_weights[c] = (
                    self.client_training_size[c] / total_training_size
                )

            self.fed_server.set_weights(self.client_weights)
            self.logger.debug(
                f"[Round: {self.round_number: 04}] self.client_weights: {self.client_weights}"
            )
            self.logger.debug(
                f"[Round: {self.round_number: 04}] self.client_training_size: {self.client_training_size}"
            )
            self.logger.debug(
                f"[Round: {self.round_number: 04}] self.fed_server.weights: {self.fed_server.weights}"
            )
            return self.client_weights[client_id]
        else:
            return -1.0

    """
    Update model weights of a global model. After updating, we increment the round number.
    """

    def update_model_weights(self):
        self.logger.info(f"[Round: {self.round_number: 04}] Updating model weights")
        self.logger.debug(
            f"[Round: {self.round_number: 04}] self.fed_server.weights: {self.fed_server.weights}"
        )
        self.fed_server.update(self.client_states)

        if self.cfg.validation == True:
            self.test_loss, self.accuracy = validation(self.fed_server, self.dataloader)

            if self.accuracy > self.best_accuracy:
                self.best_accuracy = self.accuracy
            self.logger.info(
                f"[Round: {self.round_number: 04}] Test set: Average loss: {self.test_loss:.4f}, Accuracy: {self.accuracy:.2f}%, Best Accuracy: {self.best_accuracy:.2f}%"
            )

        if (
            self.round_number % self.cfg.checkpoints_interval == 0
            or self.round_number == self.cfg.num_epochs
        ):
            """Saving model"""
            if self.cfg.save_model == True:
                save_model_iteration(self.round_number, self.model, self.cfg)

        self.round_number += 1

    """
    Check if we have received model weights from all clients for this round.
    """

    def is_round_finished(self):
        return all(
            (c, self.round_number) in self.client_learning_status
            for c in range(0, self.num_clients)
        )

    """
    Receive model weights from a client. When we have received weights from all clients,
    it will trigger a global model update.
    """

    def send_learning_results(self, client_id, round_number, penalty, primal, dual):
        self.logger.debug(
            f"[Round: {self.round_number: 04}] self.fed_server.weights: {self.fed_server.weights}"
        )
        primal_tensors = OrderedDict()
        dual_tensors = OrderedDict()
        decompress_time = 0.0
        if self.cfg.compressed_weights_client == True:
            compressor = Compressor(self.cfg)
            decompress_time_start = time.time()
            comp_model = primal[0]
            decomp_model = compressor.decompress_model(
                compressed_model=comp_model.data_bytes,
                model=self.model,
                param_count_threshold=self.cfg.param_cutoff,
            )
            for name, param in decomp_model.state_dict().items():
                primal_tensors[name] = param
            decompress_time = time.time() - decompress_time_start
        else:
            for tensor in primal:
                name = tensor.name
                shape = tuple(tensor.data_shape)
                flat = np.frombuffer(tensor.data_bytes, dtype=eval(tensor.data_dtype))
                nparray = np.reshape(flat, newshape=shape, order="C")
                primal_tensors[name] = torch.from_numpy(nparray)
        for tensor in dual:
            name = tensor.name
            shape = tuple(tensor.data_shape)
            flat = np.frombuffer(tensor.data_bytes, dtype=eval(tensor.data_dtype))
            nparray = np.reshape(flat, newshape=shape, order="C")
            dual_tensors[name] = torch.from_numpy(nparray)
        self.client_states[client_id]["primal"] = primal_tensors
        self.client_states[client_id]["dual"] = dual_tensors
        self.client_states[client_id]["penalty"] = OrderedDict()
        self.client_states[client_id]["penalty"][client_id] = penalty
        self.client_learning_status[(client_id, round_number)] = True
        self.logger.debug(
            f"[Round: {self.round_number: 04}] self.fed_server.weights: {self.fed_server.weights}"
        )

        # Round is finished when we have received model weights from all clients.
        if self.is_round_finished():
            self.logger.info(
                f"[Round: {self.round_number: 04}] Finished; all clients have sent their results."
            )
            self.update_model_weights()

        stats_file = "./data/stats_%s_%s_%s_%s_%d.csv" % (
            self.cfg.dataset,
            self.cfg.model,
            self.cfg.fed.servername,
            self.cfg.compressor,
            client_id,
        )
        with open(stats_file, "a") as f:
            f.write(
                str(decompress_time)
                + ","
                + str(0.0)
                + ","
                + str(self.round_number - 1)
                + ","
                + str(self.cfg.num_epochs)
                + ","
                + str(self.cfg.num_clients)
                + ","
                + str(0.0)
                + ","
                + str(0.0)
                + ","
                + str(0.0)
                + ","
                + str(self.test_loss)
                + ","
                + str(self.test_accuracy)
                + ","
                + str(self.best_accuracy)
                + ","
                + self.cfg.fed.servername
                + ","
                + self.cfg.fed.args.optim
                + ","
            )
