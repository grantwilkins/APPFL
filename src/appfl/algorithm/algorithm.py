import copy
from typing import Dict, Tuple
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import os
import logging


class BaseServer:
    """Abstract class of PPFL algorithm for server that aggregates and updates model parameters.

    Args:
        weight (Dict): aggregation weight assigned to each client
        model (nn.Module): torch neural network model to train
        loss_fn (nn.Module): loss function
        num_clients (int): the number of clients
        device (str): device for computation
    """

    def __init__(
        self,
        weights: OrderedDict,
        model: nn.Module,
        loss_fn: nn.Module,
        num_clients: int,
        device,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.num_clients = num_clients
        self.device = device
        self.weights = copy.deepcopy(weights)
        self.penalty = OrderedDict()
        self.prim_res = 0
        self.dual_res = 0
        self.global_state = OrderedDict()
        self.primal_states = OrderedDict()
        self.dual_states = OrderedDict()
        self.primal_states_curr = OrderedDict()
        self.primal_states_prev = OrderedDict()
        for i in range(num_clients):
            self.primal_states[i] = OrderedDict()
            self.dual_states[i] = OrderedDict()
            self.primal_states_curr[i] = OrderedDict()
            self.primal_states_prev[i] = OrderedDict()

    def get_model(self) -> nn.Module:
        """Get the model

        Return:
            nn.Module: a deepcopy of self.model
        """
        return copy.deepcopy(self.model)

    def set_weights(self, weights: OrderedDict):
        for key, value in weights.items():
            self.weights[key] = value

    def primal_recover_from_local_states(self, local_states):
        for sid, states in enumerate(local_states):
            if states is not None:
                self.primal_states[sid] = states["primal"]

    def dual_recover_from_local_states(self, local_states):
        for sid, states in enumerate(local_states):
            if states is not None:
                self.dual_states[sid] = states["dual"]

    def penalty_recover_from_local_states(self, local_states):
        for sid, states in enumerate(local_states):
            if states is not None:
                self.penalty[sid] = states["penalty"][sid]

    def primal_residual_at_server(self) -> float:
        primal_res = 0
        for i in range(self.num_clients):
            for name, _ in self.model.named_parameters():
                primal_res += torch.sum(
                    torch.square(
                        self.global_state[name].to(self.device)
                        - self.primal_states[i][name].to(self.device)
                    )
                )
        self.prim_res = torch.sqrt(primal_res).item()

    def dual_residual_at_server(self) -> float:
        dual_res = 0
        if self.is_first_iter == 1:
            for i in range(self.num_clients):
                for name, _ in self.model.named_parameters():
                    self.primal_states_curr[i][name] = copy.deepcopy(
                        self.primal_states[i][name].to(self.device)
                    )
            self.is_first_iter = 0

        else:
            self.primal_states_prev = copy.deepcopy(self.primal_states_curr)
            for i in range(self.num_clients):
                for name, _ in self.model.named_parameters():
                    self.primal_states_curr[i][name] = copy.deepcopy(
                        self.primal_states[i][name].to(self.device)
                    )

            ## compute dual residual
            for name, _ in self.model.named_parameters():
                temp = 0
                for i in range(self.num_clients):
                    temp += self.penalty[i] * (
                        self.primal_states_prev[i][name]
                        - self.primal_states_curr[i][name]
                    )

                dual_res += torch.sum(torch.square(temp))
            self.dual_res = torch.sqrt(dual_res).item()

    def log_title(self):
        title = "%10s %10s %10s %10s %10s %10s %10s %10s" % (
            "Iter",
            "Local(s)",
            "Global(s)",
            "Valid(s)",
            "PerIter(s)",
            "Elapsed(s)",
            "TestLoss",
            "TestAccu",
        )
        return title

    def log_contents(self, cfg, t):
        contents = "%10d %10.2f %10.2f %10.2f %10.2f %10.2f %10.4f %10.2f" % (
            t + 1,
            cfg["logginginfo"]["LocalUpdate_time"],
            cfg["logginginfo"]["GlobalUpdate_time"],
            cfg["logginginfo"]["Validation_time"],
            cfg["logginginfo"]["PerIter_time"],
            cfg["logginginfo"]["Elapsed_time"],
            cfg["logginginfo"]["test_loss"],
            cfg["logginginfo"]["test_accuracy"],
        )
        return contents

    def log_summary(self, cfg: DictConfig, logger):
        logger.info("Device=%s" % (cfg.device))
        logger.info("#Processors=%s" % (cfg["logginginfo"]["comm_size"]))
        logger.info("#Clients=%s" % (self.num_clients))
        logger.info("Server=%s" % (cfg.fed.servername))
        logger.info("Clients=%s" % (cfg.fed.clientname))
        logger.info("Comm_Rounds=%s" % (cfg.num_epochs))
        logger.info("Local_Rounds=%s" % (cfg.fed.args.num_local_epochs))
        logger.info("DP_Eps=%s" % (cfg.fed.args.epsilon))
        logger.info("Clipping=%s" % (cfg.fed.args.clip_value))
        logger.info("Elapsed_time=%s" % (round(cfg["logginginfo"]["Elapsed_time"], 2)))
        logger.info("BestAccuracy=%s" % (round(cfg["logginginfo"]["BestAccuracy"], 2)))


"""This implements a base class for clients."""


class BaseClient:
    """Abstract class of PPFL algorithm for client that trains local model.

    Args:
        id: unique ID for each client
        weight: aggregation weight assigned to each client
        model: (nn.Module): torch neural network model to train
        loss_fn (nn.Module): loss function
        dataloader: PyTorch data loader
        device (str): device for computation
    """

    def __init__(
        self,
        id: int,
        weight: Dict,
        model: nn.Module,
        loss_fn: nn.Module,
        dataloader: DataLoader,
        cfg,
        outfile,
        test_dataloader,
    ):
        self.id = id
        self.weight = weight
        self.model = model
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.cfg = cfg
        self.outfile = outfile
        self.test_dataloader = test_dataloader

        self.primal_state = OrderedDict()
        self.dual_state = OrderedDict()
        self.primal_state_curr = OrderedDict()
        self.primal_state_prev = OrderedDict()

    def update(self):
        """Update local model parameters"""
        raise NotImplementedError

    def get_model(self):
        """Get the model

        Return:
            the ``state_dict`` of local model
        """
        return self.model.state_dict()

    def primal_residual_at_client(self, global_state) -> float:
        primal_res = 0
        for name, _ in self.model.named_parameters():
            primal_res += torch.sum(
                torch.square(
                    global_state[name].to(self.cfg.device)
                    - self.primal_state[name].to(self.cfg.device)
                )
            )
        primal_res = torch.sqrt(primal_res).item()
        return primal_res

    def dual_residual_at_client(self) -> float:
        dual_res = 0
        if self.is_first_iter == 1:
            self.primal_state_curr = self.primal_state
            self.is_first_iter = 0

        else:
            self.primal_state_prev = self.primal_state_curr
            self.primal_state_curr = self.primal_state

            ## compute dual residual
            for name, _ in self.model.named_parameters():
                temp = self.penalty * (
                    self.primal_state_prev[name] - self.primal_state_curr[name]
                )

                dual_res += torch.sum(torch.square(temp))
            dual_res = torch.sqrt(dual_res).item()

        return dual_res

    def residual_balancing(self, prim_res, dual_res):
        if prim_res > self.residual_balancing.mu * dual_res:
            self.penalty = self.penalty * self.residual_balancing.tau
        if dual_res > self.residual_balancing.mu * prim_res:
            self.penalty = self.penalty / self.residual_balancing.tau

    def client_log_title(self):
        title = "%10s %10s %10s %10s %10s %10s %10s \n" % (
            "Round",
            "LocalEpoch",
            "PerIter[s]",
            "TrainLoss",
            "TrainAccu",
            "TestLoss",
            "TestAccu",
        )
        self.outfile.write(title)
        self.outfile.flush()

    def client_log_content(
        self, t, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
    ):
        contents = "%10s %10s %10.2f %10.4f %10.4f %10.4f %10.4f \n" % (
            self.round,
            t,
            per_iter_time,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
        )
        self.outfile.write(contents)
        self.outfile.flush()

    def client_validation(self, dataloader, metric):
        if self.loss_fn is None or dataloader is None:
            return 0.0, 0.0

        device = "cuda" if torch.cuda.is_available() else "cpu"
        validation_model = copy.deepcopy(self.model)
        validation_model.to(device)
        validation_model.eval()

        loss, tmpcnt = 0, 0
        with torch.no_grad():
            for img, target in dataloader:
                tmpcnt += 1
                img = img.to(device)
                target = target.to(device)
                output = validation_model(img)
                loss += self.loss_fn(output, target).item()
        loss = loss / tmpcnt
        accuracy = self._evaluate_model_on_tests(validation_model, dataloader, metric)
        return loss, accuracy

    def _evaluate_model_on_tests(self, model, test_dataloader, metric):
        model.eval()
        with torch.no_grad():
            test_dataloader_iterator = iter(test_dataloader)
            y_pred_final = []
            y_true_final = []
            for X, y in test_dataloader_iterator:
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                y_pred = model(X).detach().cpu()
                y = y.detach().cpu()
                y_pred_final.append(y_pred.numpy())
                y_true_final.append(y.numpy())

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            accuracy = float(metric(y_true_final, y_pred_final))
        return accuracy

    """ 
    Differential Privacy 
        (Laplacian mechanism) 
        - Noises from a Laplace dist. with zero mean and "scale_value" are added to the primal_state 
        - Variance = 2*(scale_value)^2
        - scale_value = sensitivty/epsilon, where sensitivity is determined by data, algorithm.
    """

    def laplace_mechanism_output_perturb(self, scale_value):
        """Differential privacy for output perturbation based on Laplacian distribution.
        This output perturbation adds Laplace noise to ``primal_state``.

        Args:
            scale_value: scaling vector to control the variance of Laplacian distribution
        """

        for name, param in self.model.named_parameters():
            mean = torch.zeros_like(param.data)
            scale = torch.zeros_like(param.data) + scale_value
            m = torch.distributions.laplace.Laplace(mean, scale)
            self.primal_state[name] += m.sample()
