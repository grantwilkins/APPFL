"""
This test script generates the python executions to run basic experiments of lossy 
compression and pruning on FMNIST dataset using different error bounds and server
algorithms.
"""
import numpy as np

error_bounds = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

pruning_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

num_client = 1

num_epochs = 10

num_local_epochs = 5

server_algorithm = "IIADMMServer"

federation_type = "IIADMM"

models = ["AlexNetCaltech", "MobileNetV2", "ResNet50"]

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
    for error_bound, pruning_threshold in zip(error_bounds, pruning_thresholds):
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
