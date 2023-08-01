"""
This test script generates the python executions to run basic experiments of lossy 
compression and pruning on Caltech101 dataset using different error bounds and server
algorithms.
"""
import numpy as np

error_bounds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

num_client = 1

num_epochs = 10

server_algorithm = "ServerFedAvg"

model = "AlexNetCaltech"

compressors = ["ZFP", "SZ3"]

for error_bound in error_bounds:
    for compressor in compressors:
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
            "mv ./precompress.pt ./plots/pre_%s_AlexNet_%f.pt"
            % (compressor, error_bound)
        )
        print(
            "mv ./postcompress.pt ./plots/post_%s_AlexNet_%f.pt"
            % (
                compressor,
                error_bound,
            )
        )
