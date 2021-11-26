from typing import Callable

import torch

from models.resnet import ResNet_ReID

model_list = {
    'resnet': ResNet_ReID,
}

optimizer_list = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}

scheduler_list = {
    'step_lr': torch.optim.lr_scheduler.StepLR
}


def get_callable_model(model_name: str) -> Callable:
    if model_name.lower() not in model_list.keys():
        raise ValueError(f"Could not find the model named '{model_name}'.")
    return model_list[model_name.lower()]


def get_callable_optimizer(optimizer_name: str) -> Callable:
    if optimizer_name.lower() not in optimizer_list.keys():
        raise ValueError(f"Could not find the optimizer named '{optimizer_name}'.")
    return optimizer_list[optimizer_name.lower()]


def get_callable_scheduler(scheduler_name: str) -> Callable:
    if scheduler_name.lower() not in scheduler_list.keys():
        raise ValueError(f"Could not find the scheduler named '{scheduler_name}'.")
    return scheduler_list[scheduler_name.lower()]
