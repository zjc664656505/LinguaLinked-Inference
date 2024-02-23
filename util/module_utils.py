import os
import torch
import torch.fx
from torch.fx import Graph, GraphModule, Node
from functools import wraps
from typing import Optional
from system_pipeline.model_split import model_split


def load_single_tensor(tensor_dir: str = '', map_location=torch.device('cpu')) -> torch.Tensor:
    return torch.load(tensor_dir, map_location=map_location)


def load_all_tensor(tensor_dir: str = '', map_location=torch.device('cpu')) -> list:
    files = []
    for root, directories, filenames in os.walk(tensor_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            files.append(filepath)

    return [torch.load(tensors, map_location=map_location) for tensors in files]


def save_tensor(torch_tensor: torch.Tensor = None, tensor_idx: int = 0) -> None:
    tensor_dir = "./tensor_dir/"
    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)
    torch.save(torch_tensor, tensor_dir + f"module_tensor{tensor_idx}.pt")


def add_method(cls):
    """
    Method for adding class level method in runtime dynamically

    Usage:
    class A:
        pass

    a = A()
    @add_method(A)
    def test():
        print("test")

    a.test() --> "test"
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still be used normally

    return decorator


def create_workers(world_size: int, module_split_size: int, option: str) -> list:
    """
        Input:
            1. world_size -> The total number of workers needed.
            2. module_split_size -> The size of splitting the module into.
            3. option:
                1. all: all workers including master participate training.
                2. worker_only: no master participate training
                3. multithreading: no master participate training, but world_size == module_split_size + 1

        Return:
            list of strings -> ["worker0", "worker1", ...]
    """
    if option == "all":
        assert world_size == module_split_size
        workers = []
        for idx in range(world_size):
            workers.append(f"worker{idx}")
        return workers
    if option == "worker_only":
        assert world_size == module_split_size + 1
        workers = []
        for idx in range(1, world_size):
            workers.append(f"worker{idx}")
        return workers
    if option == "multithreading":
        assert world_size == module_split_size + 1
        workers = []
        for idx in range(world_size):
            workers.append(f"worker{idx}")
        return workers

# def read_trainer(file_dir):
#     with open(file_dir, 'r') as f:
