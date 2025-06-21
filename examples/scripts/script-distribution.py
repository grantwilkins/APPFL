"""
This test script generates the python executions to run basic experiments of lossy 
compression and pruning on Caltech101 dataset using different error bounds and server
algorithms.
"""
import numpy as np

error_bounds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

pruning_thresholds = [0.1, 0.5, 0.7, 0.9, 0.95]

num_client = 1

num_epochs = 5

num_local_epochs = 5

server_algorithm = "ServerFedAvg"

model = "CNN"

compressors = ["ZFP", "SZ3"]

for error_bound, pruning_threshold in zip(error_bounds, pruning_thresholds):
    for compressor in compressors:
        print(
            "mpiexec -np %d python3 ./cifar10.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --model %s --compressor %s --pruning --pruning_threshold %f --num_local_epochs %d"
            % (
                num_client + 1,
                server_algorithm,
                error_bound,
                num_client,
                num_epochs,
                model,
                compressor,
                pruning_threshold,
                num_local_epochs,
            )
        )
        print(
            "mv ./precompress.pt ./plots/pre_%s_CNN_%s_%s.pt"
            % (
                compressor,
                str(error_bound),
                str(pruning_threshold),
            )
        )
        print(
            "mv ./postcompress.pt ./plots/post_%s_CNN_%s_%s.pt"
            % (
                compressor,
                str(error_bound),
                str(pruning_threshold),
            )
        )
        print(
            "mpiexec -np %d python3 ./cifar10.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --model %s --compressor %s"
            % (
                num_client + 1,
                server_algorithm,
                error_bound,
                num_client,
                num_epochs,
                model,
                compressor,
            )
        )
        print(
            "mv ./precompress.pt ./plots/pre_%s_CNN_%s.pt"
            % (compressor, str(error_bound))
        )
        print(
            "mv ./postcompress.pt ./plots/post_%s_CNN_%s.pt"
            % (
                compressor,
                str(error_bound),
            )
        )
