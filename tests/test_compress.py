from appfl.compressor.compressor import *
import numpy as np
from appfl.config import *
import torch
import torch.nn as nn
import math
import copy
from torchvision.models.resnet import BasicBlock, ResNet18_Weights
from torchvision import *
from torch.sparse import *
import zstd
import pickle
from scipy import sparse
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from appfl.misc.utils import flatten_primal_or_dual


def test_basic_compress(cfg: Config) -> None:
    # Create a compressor
    compressor = Compressor(cfg)

    # Create a random 1D array
    ori_data = np.random.rand(1000)
    ori_shape = ori_data.shape
    ori_dtype = ori_data.dtype

    # Compress the array
    cmpr_data_bytes = compressor.compress(ori_data=ori_data)
    cmpr_data = np.frombuffer(cmpr_data_bytes, dtype=np.uint8)
    # Decompress the array
    dec_data = compressor.decompress(
        cmp_data=cmpr_data, ori_shape=ori_shape, ori_dtype=ori_dtype
    )
    # Check if the decompressed array is the same as the original array
    (max_diff, _, _) = compressor.verify(ori_data=ori_data, dec_data=dec_data)
    assert max_diff < cfg.compressor_error_bound


def magnitude_prune(model: nn.Module, prune_ratio: float):
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
    print(nonzero_total, param_total, nonzero_total / param_total)
    return model_copy


def flatten(model: nn.Module):
    """
    Flatten the model parameters

    Args:
    model: the PyTorch model to flatten.

    Returns:
    The flattened model.
    """
    # Make a copy of the model
    model_copy = copy.deepcopy(model)
    flat_params = np.array([])
    for _, param in model_copy.named_parameters():
        param_flattened = param.detach().cpu().numpy().flatten()
        flat_params = np.concatenate((flat_params, param_flattened))
    return flat_params


def test_model_compress(cfg: Config, model: nn.Module) -> None:
    # Create a compressor
    compressor = Compressor(cfg)
    # Define the AlexNetMNIST model
    # model = resnet18(num_channel=1, num_classes=10, pretrained=1)
    # model = AlexNetMNIST(num_channel=3, num_classes=10, num_pixel=32)
    # model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model_copy = copy.deepcopy(model)
    # pruned_model = magnitude_prune(model, 0.5)
    # Flatten the model parameters)
    compressed_weights = {}
    weights_size = 0
    size = 0
    flat_og = flatten_primal_or_dual(model.state_dict())
    for name, param in model.state_dict().items():
        size += param.numel() * 4
    for p in [2**i for i in range(0, 22)]:
        (comp_model, num_lossy) = compressor.compress_model(model, p)
        print("%d Percent Lossy %f" % (p, num_lossy / flat_og.size))
        decomp_model = compressor.decompress_model(
            compressed_model=comp_model, model=model, param_count_threshold=p
        )

    """
    flat_decomp = flatten(decomp_model)
    diff = flat_og - flat_decomp
    diff = diff[diff != 0]
    sns.set(style="ticks")
    sns.set_context("paper")
    sns.histplot(diff, bins=round(math.sqrt(diff.shape[0])), kde=True)
    plt.show()
    """

    # Reasseble the model
    # Check if the reassembled model is the same shape as the original model
    for p, p_copy in zip(model.parameters(), model_copy.parameters()):
        assert p.shape == p_copy.shape


if __name__ == "__main__":
    # Config setup
    cfg = OmegaConf.structured(Config)
    cfg.compressed_weights_client = True
    cfg.compressor = "ZFP"
    cfg.lossless_compressor = "blosc"
    # cfg.compressor_lib_path = "/Users/grantwilkins/SZ3/build/sz3c/libSZ3c.dylib"
    cfg.compressor_lib_path = "/Users/grantwilkins/SZ/build/sz/libSZ.dylib"
    cfg.compressor_error_bound = 0.001
    cfg.compressor_error_mode = "REL"
    compressors = ["zstd"]
    models_test = [
        models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
    ]
    for model in models_test:
        for compressor in compressors:
            cfg.lossless_compressor = compressor
            test_model_compress(cfg=cfg, model=model)
