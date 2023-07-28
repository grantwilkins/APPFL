"""
This test script generates the python executions to run basic experiments of lossy 
compression and pruning on Caltech101 dataset using different error bounds and server
algorithms.
"""
import numpy as np

error_bound = 5e-2

num_clients = [1, 2, 4, 8, 16, 32, 64]

num_epochs = 20

server_algorithms = ["ServerFedAvg", "ServerFedAdagrad", "IIADMMServer"]

federation_type = "Federated"

pruning_threshold = 0.5

models = ["AlexNetCaltech", "resnet18"]

for server_algorithm in server_algorithms:
    if server_algorithm == "IIADMMServer":
        federation_type = "IIADMM"
    for model in models:
        for num_client in num_clients:
            print(
                "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --pruning --pruning_threshold %f --model %s --federation_type %s"
                % (
                    num_client + 1,
                    server_algorithm,
                    0.0,
                    num_client,
                    num_epochs,
                    pruning_threshold,
                    model,
                    federation_type,
                )
            )
            print(
                "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --pruning --pruning_threshold %f --model %s --federation_type %s"
                % (
                    num_client + 1,
                    server_algorithm,
                    error_bound,
                    num_client,
                    num_epochs,
                    pruning_threshold,
                    model,
                    federation_type,
                )
            )
            print(
                "mpiexec -np %d python3 ./caltech101.py --server %s --error_bound %f --num_clients %d --num_epochs %d --compressed_client --compressed_server --pruning --pruning_threshold %f --model %s --federation_type %s"
                % (
                    num_client + 1,
                    server_algorithm,
                    error_bound,
                    num_client,
                    num_epochs,
                    pruning_threshold,
                    model,
                    federation_type,
                )
            )
