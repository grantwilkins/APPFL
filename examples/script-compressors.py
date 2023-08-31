"""
This test script generates the python executions to run basic experiments of lossy 
compression and pruning on Caltech101 dataset using different error bounds and server
algorithms.
"""
import numpy as np

error_bounds = [1e-2, 1e-3, 1e-4]

num_epochs = 10

model = "AlexNetCIFAR"

compressors = [
    "SZ3",
    "SZ2",
    "ZFP",
]


for repeat in range(5):
    for compressor in compressors:
        for error_bound in error_bounds:
            print(
                "mpiexec -np %d python3 ./cifar10.py --error_bound %f --num_epochs %d --compressed_client --model %s --compressor %s"
                % (
                    2,
                    error_bound,
                    num_epochs,
                    model,
                    compressor,
                )
            )
