from typing import Any

import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from tools.logger import Logger
from tools.utils import torch_device


class OperatorModule(object):

    def __init__(self, optimizer: Optimizer, criterion: _Loss, scheduler: _LRScheduler = None,
                 device: str = None, logger: Logger = None, **kwargs):
        self.device = torch_device(device)
        self.logger = logger if logger is not None else Logger()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = kwargs

    @staticmethod
    def iter_dataloader(*dataloaders: DataLoader):
        if len(dataloaders) == 1 and isinstance(dataloaders[0], list):
            dataloaders = dataloaders[0]
        for dataloader in dataloaders:
            for value in dataloader:
                yield value

    def invoke_train(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> Any:
        raise NotImplementedError

    def _invoke_train(self, model: nn.Module, data: Any, target: Any, **kwargs) -> Any:
        raise NotImplementedError

    def invoke_predict(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> Any:
        raise NotImplementedError

    def _invoke_predict(self, model: nn.Module, data: Any, target: Any, **kwargs) -> Any:
        raise NotImplementedError

    def invoke_valid(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> Any:
        raise NotImplementedError

    def _invoke_valid(self, model: nn.Module, data: Any, target: Any, **kwargs) -> Any:
        raise NotImplementedError

    def invoke_inference(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> Any:
        raise NotImplementedError

    def _invoke_inference(self, model: nn.Module, data: Any, **kwargs) -> Any:
        raise NotImplementedError
