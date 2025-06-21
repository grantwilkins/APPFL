"""
This test script generates the python executions to run basic experiments of lossy 
compression and pruning on Caltech101 dataset using different error bounds and server
algorithms.
"""
import numpy as np

error_bound = 5e-2

num_client = 1

num_epochs = 10

server_algorithm = "ServerFedAvg"

pruning_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

model = "AlexNetCaltech"

for pruning_threshold in pruning_thresholds:
    print(
        "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --model %s"
        % (num_client + 1, server_algorithm, 0.0, num_client, num_epochs, model)
    )
    print(
        "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --pruning --pruning_threshold %f --model %s"
        % (
            num_client + 1,
            server_algorithm,
            0.0,
            num_client,
            num_epochs,
            pruning_threshold,
            model,
        )
    )
    print(
        "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --pruning --pruning_threshold %f --model %s"
        % (
            num_client + 1,
            server_algorithm,
            error_bound,
            num_client,
            num_epochs,
            pruning_threshold,
            model,
        )
    )
    print(
        "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --compressed_server --pruning --pruning_threshold %f --model %s"
        % (
            num_client + 1,
            server_algorithm,
            error_bound,
            num_client,
            num_epochs,
            pruning_threshold,
            model,
        )
    )
