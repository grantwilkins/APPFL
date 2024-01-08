error_bound = 1e-2
num_epochs = 10
model = "MobileNetV2"
cutoff_thresholds = [2**i for i in range(0, 22)]

for rep in range(5):
    print("mpiexec -np 2 python3 cifar10.py --model MobileNetV2 --num_epochs 10")
    for cutoff in cutoff_thresholds:
        print(
            "mpiexec -np 2 python3 cifar10.py --compressed_client --compressor SZ2 --param_cutoff %d --error_bound %f --num_epochs %d --model %s"
            % (cutoff, error_bound, num_epochs, model)
        )
