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


def test_model_compress(cfg: Config, model: nn.Module) -> None:
    # Create a compressor
    compressor = Compressor(cfg)
    compressor.compress_model(model=model)


if __name__ == "__main__":
    # Config setup
    cfg = OmegaConf.structured(Config)
    cfg.compressed_weights_client = True
    cfg.compressor = "SZ2"
    cfg.lossless_compressor = "blosc"
    # cfg.compressor_lib_path = "/Users/grantwilkins/SZ3/build/sz3c/libSZ3c.dylib"
    # cfg.compressor_lib_path = "/Users/grantwilkins/SZ/build/sz/libSZ.dylib"
    cfg.compressor_error_bound = 0.5
    cfg.compressor_error_mode = "REL"
    compressors = ["SZ3", "SZ2", "ZFP"]
    models_test = [
        models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
        models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    ]
    for model in models_test:
        for compressor in compressors:
            cfg.compressor = compressor
            test_model_compress(cfg=cfg, model=model)
