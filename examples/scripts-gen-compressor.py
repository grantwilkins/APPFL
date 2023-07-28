"""
This test script generates the python executions to run basic experiments of lossy 
compression and pruning on Caltech101 dataset using different error bounds and server
algorithms.
"""
import numpy as np

error_bound = [1e-2, 1e-3, 1e-4]

num_client = 5

num_epochs = 20

server_algorithm = "ServerFedAvg"

pruning_threshold = 0.5

model = "AlexNetCaltech"

compressors = ["SZ3", "SZx", "ZFP", "Prune"]


for compressor in compressors:
    print(
        "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --pruning --pruning_threshold %f --model %s --compressor %s"
        % (
            num_client + 1,
            server_algorithm,
            error_bound,
            num_client,
            num_epochs,
            pruning_threshold,
            model,
            compressor,
        )
    )
    print(
        "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --compressed_server --pruning --pruning_threshold %f --model %s --compressor %s"
        % (
            num_client + 1,
            server_algorithm,
            error_bound,
            num_client,
            num_epochs,
            pruning_threshold,
            model,
            compressor,
        )
    )
