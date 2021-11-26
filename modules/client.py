import os
from typing import Dict, Any, Union, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.operator import OperatorModule
from tools.logger import Logger
from tools.utils import torch_device


class ClientModule(object):
    def __init__(
            self,
            client_name: str,
            model: nn.Module,
            operator: OperatorModule,
            ckpt_root: str,
            model_ckpt_name: str = None,
            **kwargs
    ):
        self.device = torch_device(**kwargs)
        self.client_name = client_name
        self.model = model
        self.operator = operator
        self.args = kwargs
        self.ckpt_path = os.path.join(ckpt_root, self.client_name)
        self.model_ckpt_name = model_ckpt_name
        self.logger = Logger(f'{client_name}')
        self.operator.logger = self.logger
        self.logger.info('Startup successfully.')

    def load_state(
            self,
            state_name: str,
            default_value: torch.Tensor = None
    ) -> torch.Tensor:
        state_path = os.path.join(self.ckpt_path, f'{state_name}.ckpt')
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if os.path.exists(state_path):
            return torch.load(state_path)
        elif default_value is not None:
            return default_value
        else:
            raise ValueError(f"State checkpoint does not exist in '{state_path}'.")

    def save_state(
            self,
            state_name: str,
            state: torch.Tensor,
            cover: bool = False
    ) -> Any:
        state_path = os.path.join(self.ckpt_path, f'{state_name}.ckpt')
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if cover is False and os.path.exists(state_path):
            raise ValueError(f"State checkpoint has already exist in '{state_path}'.")
        torch.save(state, state_path)

    def load_model(self, model_name: str):
        self.model.load_state_dict(self.load_state(
            state_name=model_name,
            default_value=self.model.state_dict()
        ))

    def save_model(self, model_name: str):
        self.save_state(model_name, self.model.state_dict(), True)

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        model_dict = self.model.state_dict()
        for n, p in params_state.items():
            model_dict[n] = p.clone().detach()
        self.model.load_state_dict(model_dict)

    def get_incremental_state(self, **kwargs) -> Dict:
        return None

    def get_integrated_state(self, **kwargs) -> Dict:
        return None

    def update_by_incremental_state(self, state: Dict, **kwargs) -> Any:
        return None

    def update_by_integrated_state(self, state: Dict, **kwargs) -> Any:
        return None

    def train(
            self,
            epochs: int,
            task_name: str,
            tr_loader: Union[List[DataLoader], DataLoader],
            val_loader: Union[List[DataLoader], DataLoader],
            **kwargs
    ) -> Any:
        raise NotImplementedError

    def train_one_epoch(
            self,
            task_name: str,
            tr_loader: Union[List[DataLoader], DataLoader],
            val_loader: Union[List[DataLoader], DataLoader],
            **kwargs
    ) -> Any:
        raise NotImplementedError

    def inference(
            self,
            task_name: str,
            query_loader: Union[List[DataLoader], DataLoader],
            gallery_loader: Union[List[DataLoader], DataLoader],
            **kwargs
    ) -> Any:
        raise NotImplementedError

    def validate(
            self,
            task_name: str,
            query_loader: Union[List[DataLoader], DataLoader],
            gallery_loader: Union[List[DataLoader], DataLoader],
            **kwargs
    ) -> Any:
        raise NotImplementedError
