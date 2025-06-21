models = ["AlexNetMNIST", "MobileNetV2", "ResNet50"]
for i in range(5):
    for model in models:
        print("mpiexec -np 2 python3 ./fmnist.py --error_bound 0.0 --num_clients 1 --num_epochs 10 --model %s" % (model))

models = ["AlexNetCaltech", "MobileNetV2", "ResNet50"]
for i in range(5):
    for model in models:
        print("mpiexec -np 2 python3 ./caltech101.py --error_bound 0.0 --num_clients 1 --num_epochs 10 --model %s" % (model))

models = ["AlexNetCIFAR", "ResNet50"]
for i in range(5):
    for model in models:
        print("mpiexec -np 2 python3 ./cifar10.py --error_bound 0.0 --num_clients 1 --num_epochs 10 --model %s" % (model))