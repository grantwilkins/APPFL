from typing import OrderedDict
from collections import OrderedDict
import torch
import os
from omegaconf import DictConfig
import logging
import random
import numpy as np
import copy
import copy


def validation(self, dataloader, metric):
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
    accuracy = _evaluate_model_on_tests(validation_model, dataloader, metric)
    return loss, accuracy


def _evaluate_model_on_tests(model, test_dataloader, metric):
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


def create_custom_logger(logger, cfg: DictConfig):
    dir = cfg.output_dirname
    if os.path.isdir(dir) == False:
        os.mkdir(dir)
    output_filename = cfg.output_filename + "_server"

    file_ext = ".txt"
    filename = dir + "/%s%s" % (output_filename, file_ext)
    uniq = 1
    while os.path.exists(filename):
        filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
        uniq += 1

    logger.setLevel(logging.INFO)
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def client_log(dir, output_filename):
    if os.path.isdir(dir) == False:
        os.mkdir(dir)

    file_ext = ".txt"
    filename = dir + "/%s%s" % (output_filename, file_ext)
    uniq = 1
    while os.path.exists(filename):
        filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
        uniq += 1

    outfile = open(filename, "a")

    return outfile


def load_model(cfg: DictConfig):
    file = cfg.load_model_dirname + "/%s%s" % (cfg.load_model_filename, ".pt")
    model = torch.load(file)
    model.eval()
    return model


def save_model_iteration(t, model, cfg: DictConfig):
    dir = cfg.save_model_dirname
    if os.path.isdir(dir) == False:
        os.mkdir(dir)

    file_ext = ".pt"
    file = dir + "/%s_Round_%s%s" % (cfg.save_model_filename, t, file_ext)
    uniq = 1
    while os.path.exists(file):
        file = dir + "/%s_Round_%s_%d%s" % (cfg.save_model_filename, t, uniq, file_ext)
        uniq += 1

    torch.save(model, file)


def set_seed(seed=233):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unflatten_model_params(model: torch.nn.Module, flat_params: np.ndarray):
    # Convert flat_params to a PyTorch tensor
    flat_params_tensor = torch.from_numpy(flat_params)

    # Make a copy of the model
    model_copy = copy.deepcopy(model)

    # Initialize a pointer variable to 0
    pointer = 0

    # Iterate over each parameter in the model
    for _, param in model_copy.named_parameters():
        # Determine the number of elements in the parameter
        num_elements = param.numel()

        # Slice that number of elements from the flat_params_tensor using the pointer variable
        param_slice = flat_params_tensor[pointer : pointer + num_elements]

        # Reshape the resulting slice to match the shape of the parameter
        reshaped_slice = param_slice.view(param.shape)

        # Assign the reshaped_slice back to the corresponding parameter in the model
        param.data = reshaped_slice

        # Increment the pointer variable by the number of elements used
        pointer += num_elements

    # Return the state_dict of the modified model
    return model_copy.state_dict()


def flatten_model_params(model: torch.nn.Module) -> np.ndarray:
    # Concatenate all of the tensors in the model's state_dict into a 1D tensor
    flat_params = torch.cat([param.view(-1) for _, param in model.named_parameters()])
    # Convert the tensor to a numpy array and return it
    return flat_params.detach().cpu().numpy()
