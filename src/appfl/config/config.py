from dataclasses import dataclass, field
from typing import Any, Tuple
from omegaconf import DictConfig, OmegaConf


from .fed.federated import *
from .fed.fedasync import *
from .fed.iceadmm import *  ## TODO: combine iceadmm and iiadmm under the name of ADMM.
from .fed.iiadmm import *
import numpy as np


@dataclass
class Config:
    fed: Any = field(default_factory=Federated)

    # Compute device
    device: str = "mps"
    device_server: str = "mps"

    # Model
    model: str = "CNN"

    # Dataset
    dataset: str = "MNIST"

    # Number of training epochs
    num_clients: int = 1

    # Number of training epochs
    num_epochs: int = 2

    # Number of workers in DataLoader
    num_workers: int = 0

    # Train data batch info
    batch_training: bool = True  ## TODO: revisit
    train_data_batch_size: int = 64
    train_data_shuffle: bool = False

    # Indication of whether to validate or not using testing data
    validation: bool = True
    test_data_batch_size: int = 64
    test_data_shuffle: bool = False

    # Checking data sanity
    data_sanity: bool = False

    # Reproducibility
    reproduce: bool = True

    # PCA on Trajectory
    pca_dir: str = ""
    params_start: int = 0
    params_end: int = 49
    ncomponents: int = 40

    # Tensorboard
    use_tensorboard: bool = False

    # Loading models
    load_model: bool = False
    load_model_dirname: str = ""
    load_model_filename: str = ""

    # Saving models (server)
    save_model: bool = False
    save_model_dirname: str = ""
    save_model_filename: str = ""
    checkpoints_interval: int = 2

    # Saving state_dict (clients)
    save_model_state_dict: bool = False

    # Logging and recording outputs
    output_dirname: str = "output"
    output_filename: str = "result"

    logginginfo: DictConfig = OmegaConf.create({})
    summary_file: str = ""

    #
    # gRPC configutations
    #

    # 100 MB for gRPC maximum message size
    max_message_size: int = 104857600

    operator: DictConfig = OmegaConf.create({"id": 1})
    server: DictConfig = OmegaConf.create(
        {"id": 1, "host": "localhost", "port": 50051, "use_tls": False, "api_key": None}
    )
    client: DictConfig = OmegaConf.create({"id": 1})

    # Compression Information
    compressed_weights_client: bool = False
    compressed_weights_server: bool = False
    compressor: str = ""
    compressor_lib_path: str = ""
    compressor_error_mode: str = ""
    compressor_error_bound: float = 0.0
    flat_model_size: Tuple[int, ...] = (0,)
    flat_model_dtype: str = "np.float32"
    pruning: bool = False
    pruning_threshold: float = 0.01
