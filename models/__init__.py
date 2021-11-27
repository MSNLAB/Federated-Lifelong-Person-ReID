from typing import Callable

import torch

from models.resnet import ResNet_ReID

net_list = {
    'resnet': ResNet_ReID,
}

optimizer_list = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}

scheduler_list = {
    'step_lr': torch.optim.lr_scheduler.StepLR
}


def get_net_constructor(net_name: str) -> Callable:
    if net_name.lower() not in net_list.keys():
        raise ValueError(f"Could not find the model net named '{net_name}'.")
    return net_list[net_name.lower()]


def get_optimizer_constructor(optimizer_name: str) -> Callable:
    if optimizer_name.lower() not in optimizer_list.keys():
        raise ValueError(f"Could not find the optimizer named '{optimizer_name}'.")
    return optimizer_list[optimizer_name.lower()]


def get_scheduler_constructor(scheduler_name: str) -> Callable:
    if scheduler_name.lower() not in scheduler_list.keys():
        raise ValueError(f"Could not find the scheduler named '{scheduler_name}'.")
    return scheduler_list[scheduler_name.lower()]
