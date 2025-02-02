import os
import time
import torch
import argparse
import torchvision
import numpy as np
import appfl.run_mpi_async as rma
import torchvision.transforms as transforms
from mpi4py import MPI
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.utils import get_model 
 
""" read arguments """ 

parser = argparse.ArgumentParser() 

parser.add_argument('--device', type=str, default="cpu")    

## dataset and model
parser.add_argument('--dataset', type=str, default="CIFAR10")   
parser.add_argument('--num_channel', type=int, default=3)   
parser.add_argument('--num_classes', type=int, default=10)   
parser.add_argument('--num_pixel', type=int, default=32)   
parser.add_argument('--model', type=str, default="CNN")   
parser.add_argument('--train_data_batch_size', type=int, default=128)   
parser.add_argument('--test_data_batch_size', type=int, default=128)   


## clients
# parser.add_argument('--num_clients', type=int, default=1)    
parser.add_argument('--client_optimizer', type=str, default="Adam")    
parser.add_argument('--client_lr', type=float, default=1e-3)    
parser.add_argument('--num_local_epochs', type=int, default=5)    

## server
parser.add_argument('--server', type=str, default="ServerFedAsynchronous")    
parser.add_argument('--num_epochs', type=int, default=20)    
parser.add_argument('--server_lr', type=float, required=False)    
parser.add_argument('--mparam_1', type=float, required=False)    
parser.add_argument('--mparam_2', type=float, required=False)    
parser.add_argument('--adapt_param', type=float, required=False)  

## Fed Async
parser.add_argument("--gradient_based", type=bool, default=False, help="Whether the algorithm requires gradient from the model")
parser.add_argument("--alpha", type=float, default=0.9, help="Mixing parameter for FedAsync Algorithm")
parser.add_argument("--staleness_func", type=str, choices=['constant', 'polynomial', 'hinge'], default='polynomial')
parser.add_argument("--a", type=float, default=0.5, help="First parameter for the staleness function")
parser.add_argument("--b", type=int, default=4, help="Second parameter for Hinge staleness function")
parser.add_argument("--K", type=int, default=3, help="Buffer size for FedBuffer algorithm")
 
args = parser.parse_args()    

args.save_model_state_dict = False

if torch.cuda.is_available():
    args.device="cuda"


def get_data(): 
    dir = os.getcwd() + "/datasets/RawData"
 
    # test data for a server
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_data_raw = eval("torchvision.datasets." + args.dataset)(
        dir, download=True, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    )

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + args.dataset)(
        dir, download=False, train=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),                        
                        normalize,
                    ])
    )

    split_train_data_raw = np.array_split(range(len(train_data_raw)), args.num_clients)
    train_datasets = []
    for i in range(args.num_clients):

        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        ) 
    return train_datasets, test_dataset

## Run
def main():
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size() 

    assert comm_size > 1, "This script requires the toal number of processes to be greater than one!"
    args.num_clients = comm_size - 1

    ## Reproducibility
    set_seed(1)

    """ Configuration """     
    cfg = OmegaConf.structured(Config(fed=FedAsync())) 

    cfg.device = args.device
    cfg.save_model_state_dict = args.save_model_state_dict

    ## dataset
    cfg.train_data_batch_size = args.train_data_batch_size
    cfg.test_data_batch_size = args.test_data_batch_size
    cfg.train_data_shuffle = True

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    
    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs        
    cfg.output_dirname = "./outputs_%s_%s_%s_%s_%s_%s"%(args.dataset, args.model, args.server, args.client_optimizer, args.num_local_epochs, args.client_lr)    
    if args.server_lr != None:
        cfg.fed.args.server_learning_rate = args.server_lr
        cfg.output_dirname += "_ServerLR_%s" %(args.server_lr)
        
    if args.adapt_param != None:
        cfg.fed.args.server_adapt_param = args.adapt_param   
        cfg.output_dirname += "_AdaptParam_%s" %(args.adapt_param)             
        
    if args.mparam_1 != None:
        cfg.fed.args.server_momentum_param_1 = args.mparam_1
        cfg.output_dirname += "_MParam1_%s" %(args.mparam_1)
        
    if args.mparam_2 != None:
        cfg.fed.args.server_momentum_param_2 = args.mparam_2  
        cfg.output_dirname += "_MParam2_%s" %(args.mparam_2)        

    cfg.output_filename = "result"          

    ## fed async/fed buffer
    cfg.fed.args.K = args.K
    cfg.fed.args.alpha = args.alpha
    cfg.fed.args.gradient_based = args.gradient_based
    cfg.fed.args.staleness_func.name = args.staleness_func
    cfg.fed.args.staleness_func.args.a = args.a
    cfg.fed.args.staleness_func.args.b = args.b

    start_time = time.time()

    """ User-defined model """    
    model = get_model(args) 
    loss_fn = torch.nn.CrossEntropyLoss()   

    ## loading models 
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname      = "./save_models"
        cfg.load_model_filename     = "Model"               
        model = load_model(cfg)         
    
    """ User-defined data """
    train_datasets, test_dataset = get_data()

    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(train_datasets, test_dataset, args.num_channel, args.num_pixel)        

    print(
        "-------Loading_Time=",
        time.time() - start_time,
    ) 
    
    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "Model" 

    """ Running """
    if comm_rank == 0:
        rma.run_server(
            cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset
        )
    else:
        assert comm_size == args.num_clients + 1
        rma.run_client(
            cfg, comm, model, loss_fn, args.num_clients, train_datasets, test_dataset
        )
    print("------DONE------", comm_rank)      
  

if __name__ == "__main__":
    main()

 
# To run CUDA-aware MPI with n clients:
# mpiexec -np n+1 --mca opal_cuda_support 1 python ./cifar10_async_mpi.py
# To run MPI with n clients:
# mpiexec -np n+1 python ./cifar10_async_mpi.py
