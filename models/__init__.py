import torch

from models.resnet import *
from models.swin_transformer import *

nets = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'swin_transformer_tiny': swin_transformer_tiny,
    'swin_transformer_small': swin_transformer_small,
    'swin_transformer_base': swin_transformer_base,
    'swin_transformer_large': swin_transformer_large,
}

optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}

schedulers = {
    'step_lr': torch.optim.lr_scheduler.StepLR
}
