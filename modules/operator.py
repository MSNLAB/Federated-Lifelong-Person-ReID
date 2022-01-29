from typing import Any

from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from modules.model import ModelModule
from tools.logger import Logger


class OperatorModule(object):

    def __init__(self, optimizer: Optimizer, criterion: _Loss, scheduler: _LRScheduler = None,
                 logger: Logger = None, **kwargs):
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

    def invoke_train(self, model: ModelModule, dataloader: DataLoader, **kwargs) -> Any:
        raise NotImplementedError

    def _invoke_train(self, model: ModelModule, data: Any, target: Any, **kwargs) -> Any:
        raise NotImplementedError

    def invoke_predict(self, model: ModelModule, dataloader: DataLoader, **kwargs) -> Any:
        raise NotImplementedError

    def _invoke_predict(self, model: ModelModule, data: Any, target: Any, **kwargs) -> Any:
        raise NotImplementedError

    def invoke_valid(self, model: ModelModule, dataloader: DataLoader, **kwargs) -> Any:
        raise NotImplementedError

    def _invoke_valid(self, model: ModelModule, data: Any, target: Any, **kwargs) -> Any:
        raise NotImplementedError

    def invoke_inference(self, model: ModelModule, dataloader: DataLoader, **kwargs) -> Any:
        raise NotImplementedError

    def _invoke_inference(self, model: ModelModule, data: Any, **kwargs) -> Any:
        raise NotImplementedError
