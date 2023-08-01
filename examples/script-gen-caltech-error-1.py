"""
This test script generates the python executions to run basic experiments of lossy 
compression and pruning on Caltech101 dataset using different error bounds and server
algorithms.
"""
import numpy as np

error_bounds = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

num_client = 2

num_epochs = 20

num_local_epochs = 5

server_algorithm = "ServerFedAvg"

federation_type = "Federated"

pruning_threshold = 0.5

models = ["AlexNetCaltech", "VGG16CIFAR", "resnet18"]

for model in models:
    print(
        "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --model %s --federation_type %s --num_local_epochs %d"
        % (
            num_client + 1,
            server_algorithm,
            0.0,
            num_client,
            num_epochs,
            model,
            federation_type,
            num_local_epochs,
        )
    )
    print(
        "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --pruning --pruning_threshold %f --model %s --federation_type %s --num_local_epochs %d"
        % (
            num_client + 1,
            server_algorithm,
            0.0,
            num_client,
            num_epochs,
            pruning_threshold,
            model,
            federation_type,
            num_local_epochs,
        )
    )
    for error_bound in error_bounds:
        print(
            "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --pruning --pruning_threshold %f --model %s --federation_type %s --num_local_epochs %d"
            % (
                num_client + 1,
                server_algorithm,
                error_bound,
                num_client,
                num_epochs,
                pruning_threshold,
                model,
                federation_type,
                num_local_epochs,
            )
        )
        print(
            "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --compressed_server --pruning --pruning_threshold %f --model %s --federation_type %s --num_local_epochs %d"
            % (
                num_client + 1,
                server_algorithm,
                error_bound,
                num_client,
                num_epochs,
                pruning_threshold,
                model,
                federation_type,
                num_local_epochs,
            )
        )
