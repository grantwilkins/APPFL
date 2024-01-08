"""
This test script generates the python executions to run basic experiments of lossy 
compression and pruning on CIFAR-10 dataset using different error bounds and server
algorithms.
"""
import numpy as np

error_bounds = [1e-1, 1e-2,  1e-3, 1e-4, 1e-5]

num_client = 1

num_epochs = 25

num_local_epochs = 1

server_algorithm = "ServerFedAvg"

federation_type = "Federated"

models = ["AlexNetCIFAR", "MobileNetV2", "ResNet50"]

for model in models:
    print(
        "mpiexec -np %d python3 ./cifar10.py --server %s --error_bound %f --num_clients %d --num_epochs %d --model %s --federation_type %s --num_local_epochs %d"
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
    for error_bound in error_bounds:
        print(
            "mpiexec -np %d python3 ./cifar10.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --model %s --federation_type %s --num_local_epochs %d --compressor SZ2"
            % (
                num_client + 1,
                server_algorithm,
                error_bound,
                num_client,
                num_epochs,
                model,
                federation_type,
                num_local_epochs,
            )
        )
