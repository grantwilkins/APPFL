model = "MobileNetV2"

num_clients = [2, 4, 8, 16, 32, 64, 128]

error_bounds = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# Weak Scaling
for num in num_clients:
    for error_bound in error_bounds:
        print(
            "mpiexec -np %d python3 ./cifar10.py --num_client %d --error_bound %f --compressed_client --federation_type Federated --num_epochs 5 --model %s"
            % (num, num - 1, error_bound, model)
        )

# Strong Scaling
for num in num_clients:
    for error_bound in error_bounds:
        print(
            "mpiexec -np %d python3 ./cifar10.py --num_client 127 --error_bound %f --compressed_client --federation_type Federated --num_epochs 5 --model %s"
            % (num, error_bound, model)
        )
