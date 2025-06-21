#!/bin/zsh

for error_bound in $(seq 0.15 -0.005 0.01)
do
   mpiexec -np 10 python3 ./mnist.py --error-bound $error_bound --num_clients 9 --server ServerFedAvg
done
