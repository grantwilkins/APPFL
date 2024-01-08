for _ in range(5):
	print("mpiexec -np 2 python3 ./cifar10.py --model MobileNetV2 --num_clients 1 --error_bound 0.0 --epochs 10")
