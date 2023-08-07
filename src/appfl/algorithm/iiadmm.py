import logging
import time

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy
import math
from appfl.misc.utils import *
from appfl.compressor import *
import torch.nn as nn


class IIADMMServer(BaseServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        super(IIADMMServer, self).__init__(weights, model, loss_fn, num_clients, device)

        self.__dict__.update(kwargs)

        self.is_first_iter = 1

        """
        At initial, dual_state = 0
        """
        for i in range(num_clients):
            for name, param in model.named_parameters():
                self.dual_states[i][name] = torch.zeros_like(param.data)

    def update(self, local_states: OrderedDict):
        """Inputs for the global model update"""
        for name, param in self.model.named_parameters():
            self.global_state[name] = param.data
        # self.global_state = copy.deepcopy(self.model.state_dict())
        super(IIADMMServer, self).primal_recover_from_local_states(local_states)
        super(IIADMMServer, self).penalty_recover_from_local_states(local_states)

        """ residual calculation """
        super(IIADMMServer, self).primal_residual_at_server()
        super(IIADMMServer, self).dual_residual_at_server()

        """ global_state calculation """
        for name, param in self.model.named_parameters():
            tmp = 0.0
            for i in range(self.num_clients):
                ## change device
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )

                ## dual
                self.dual_states[i][name] = self.dual_states[i][name].to(
                    self.device
                ) + self.penalty[i] * (
                    self.global_state[name] - self.primal_states[i][name]
                )

                ## computation
                epsilon = 1e-8  # Define a small constant to prevent division by zero
                tmp += (
                    self.primal_states[i][name]
                    - (1.0 / (self.penalty[i] + epsilon)) * self.dual_states[i][name]
                )

            self.global_state[name] = tmp / self.num_clients

        """ model update """
        # self.model.load_state_dict(self.global_state)
        for name, param in self.model.named_parameters():
            param.data = self.global_state[name]

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(IIADMMServer, self).log_title()
            title = title + "%12s %12s %12s %12s" % (
                "PrimRes",
                "DualRes",
                "RhoMin",
                "RhoMax",
            )
            logger.info(title)

        contents = super(IIADMMServer, self).log_contents(cfg, t)
        contents = contents + "%12.4e %12.4e %12.4e %12.4e" % (
            self.prim_res,
            self.dual_res,
            min(self.penalty.values()),
            max(self.penalty.values()),
        )
        logger.info(contents)

    def logging_summary(self, cfg, logger):
        super(IIADMMServer, self).log_summary(cfg, logger)


class IIADMMClient(BaseClient):
    def __init__(
        self,
        id,
        weight,
        model,
        loss_fn,
        dataloader,
        cfg,
        outfile,
        test_dataloader,
        **kwargs
    ):
        super(IIADMMClient, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        """
        At initial, (1) primal_state = global_state, (2) dual_state = 0
        """
        self.model.to(self.cfg.device)
        for name, param in model.named_parameters():
            self.primal_state[name] = param.data
            self.dual_state[name] = torch.zeros_like(param.data)

        self.penalty = kwargs["init_penalty"]
        self.is_first_iter = 1
        self.compressor = Compressor(cfg)

    def magnitude_prune(self, model: nn.Module, prune_ratio: float):
        """
        Perform magnitude-based pruning on a PyTorch model.

        Args:
        model: the PyTorch model to prune.
        prune_ratio: the percentage of weights to prune in each layer.

        Returns:
        The pruned model.
        """
        # Make a copy of the model
        model_copy = copy.deepcopy(model)
        nonzero_total = 0
        param_total = 0
        for _, param in model_copy.named_parameters():
            if param.requires_grad:
                # Flatten the tensor to 1D for easier percentile calculation
                param_flattened = param.detach().cpu().numpy().flatten()
                # Compute the threshold as the (prune_ratio * 100) percentile of the absolute values
                threshold = np.percentile(np.abs(param_flattened), prune_ratio * 100)
                # Create a mask that will be True for the weights to keep and False for the weights to prune
                mask = torch.abs(param) > threshold
                # Apply the mask
                param.data.mul_(mask.float())
                nonzero_total += torch.count_nonzero(param.data).item()
                param_total += param.data.numel()
        return model_copy

    def update(self):
        self.model.train()
        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Inputs for the local model update """
        global_state = copy.deepcopy(self.model.state_dict())

        """ Adaptive Penalty (Residual Balancing) """
        if self.residual_balancing.res_on == True:
            prim_res = super(IIADMMClient, self).primal_residual_at_client(global_state)
            dual_res = super(IIADMMClient, self).dual_residual_at_client()
            super(IIADMMClient, self).residual_balancing(prim_res, dual_res)

        """ Multiple local update """
        for i in range(self.num_local_epochs):
            for data, target in self.dataloader:
                for name, param in self.model.named_parameters():
                    param.data = self.primal_state[name].to(self.cfg.device)

                if (
                    self.residual_balancing.res_on == True
                    and self.residual_balancing.res_on_every_update == True
                ):
                    prim_res = super(IIADMMClient, self).primal_residual_at_client(
                        global_state
                    )
                    dual_res = super(IIADMMClient, self).dual_residual_at_client()
                    super(IIADMMClient, self).residual_balancing(prim_res, dual_res)

                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)

                if self.accum_grad == False:
                    optimizer.zero_grad()

                output = self.model(data)

                loss = self.loss_fn(output, target)
                loss.backward()

                if self.clip_value != False:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )

                ## STEP: Update primal
                coefficient = 1
                if self.coeff_grad == True:
                    coefficient = (
                        self.weight * len(target) / len(self.dataloader.dataset)
                    )

                self.iiadmm_step(coefficient, global_state, optimizer)

        ## Update dual
        for name, param in self.model.named_parameters():
            self.dual_state[name] = self.dual_state[name] + self.penalty * (
                global_state[name] - self.primal_state[name]
            )

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value / self.penalty
            scale_value = sensitivity / self.epsilon
            super(IIADMMClient, self).laplace_mechanism_output_perturb(scale_value)

        """
        ## store data in cpu before sending it to server
        if self.cfg.device == "cuda":
            for name, param in self.model.named_parameters():
                self.primal_state[name] = param.data.cpu()
        """

        """
        Pruning step
        """
        if self.cfg.pruning == True:
            self.model = self.magnitude_prune(self.model, self.cfg.pruning_threshold)
        # POSSIBLY COMPRESS AND LOG THE STATS
        start_time = time.time()
        compression_ratio = 0
        flat_primal = flatten_primal_or_dual(self.primal_state)
        flat_dual = flatten_primal_or_dual(self.dual_state)
        if self.cfg.compressed_weights_client == True:
            compressed_primal = self.compressor.compress(ori_data=flat_primal)
            compressed_dual = self.compressor.compress(ori_data=flat_dual)
            self.cfg.flat_model_size = flat_primal.shape
            compression_ratio = (len(flat_primal) * 8) / (
                len(compressed_dual) + len(compressed_primal)
            )
        compress_time = time.time() - start_time

        stats_file = "./data/stats_%s_%s_%s_%s_%d.csv" % (
            self.cfg.dataset,
            self.cfg.model,
            self.cfg.fed.servername,
            self.cfg.compressor,
            self.id,
        )
        with open(stats_file, "a") as f:
            f.write(
                str(self.cfg.dataset)
                + ","
                + str(self.cfg.model)
                + ","
                + str(self.id)
                + ","
                + str(self.cfg.compressed_weights_client)
                + ","
                + str(self.cfg.compressed_weights_server)
                + ","
                + str(compression_ratio)
                + ","
                + self.cfg.compressor
                + ","
                + self.cfg.compressor_error_mode
                + ","
                + str(self.cfg.compressor_error_bound)
                + ","
                + str(flat_primal.shape[0])
                + ","
                + str(compress_time)
                + ","
                + str(self.cfg.pruning)
                + ","
                + str(self.cfg.pruning_threshold)
                + ","
                + str(np.std(flat_primal))
                + ","
                + str(np.median(flat_primal))
                + ","
                + str(np.mean(flat_primal))
                + ","
            )
        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.primal_state
        if self.cfg.compressed_weights_client == True:
            self.local_state["primal"] = compressed_primal
        self.local_state["dual"] = self.dual_state
        if self.cfg.compressed_weights_client == True:
            self.local_state["dual"] = compressed_dual
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = self.penalty

        return self.local_state

    def iiadmm_step(self, coefficient, global_state, optimizer):
        momentum = 0
        if "momentum" in self.optim_args.keys():
            momentum = self.optim_args.momentum
        weight_decay = 0
        if "weight_decay" in self.optim_args.keys():
            weight_decay = self.optim_args.weight_decay
        dampening = 0
        if "dampening" in self.optim_args.keys():
            dampening = self.optim_args.dampening
        nesterov = False

        for name, param in self.model.named_parameters():
            grad = copy.deepcopy(param.grad * coefficient)

            if weight_decay != 0:
                grad.add_(weight_decay, self.primal_state[name])
            if momentum != 0:
                param_state = optimizer.state[param]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = grad.clone()
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(1 - dampening, grad)
                if nesterov:
                    grad = self.grad[name].add(momentum, buf)
                else:
                    grad = buf

            ## Update primal
            self.primal_state[name] = global_state[name] + (1.0 / self.penalty) * (
                self.dual_state[name] - grad
            )
