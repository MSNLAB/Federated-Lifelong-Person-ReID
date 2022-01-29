import os
from typing import Dict, Any

import torch

from modules.model import ModelModule
from modules.operator import OperatorModule
from tools.logger import Logger


class ServerModule(object):
    def __init__(
            self,
            server_name: str,
            model: ModelModule,
            operator: OperatorModule,
            ckpt_root: str,
            **kwargs
    ):
        self.server_name = server_name
        self.model = model
        self.operator = operator
        for n, p in kwargs.items():
            self.__setattr__(n, p)
        self.ckpt_path = os.path.join(ckpt_root, self.server_name)
        self.clients = {}
        self.logger = Logger(self.server_name)
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

    def register_client(self, client_name: str) -> bool:
        if client_name in self.clients.keys():
            self.logger.warn(f"'{client_name}' has already registered in server.")
            return False
        else:
            self.clients[client_name] = self.init_client_state()
            self.logger.info(f"'{client_name}' register succeed in server.")
            return True

    def unregister_client(self, client_name) -> bool:
        if client_name in self.clients:
            self.clients.pop(client_name)
            self.logger.info(f"'{client_name}' unregister succeed in server.")
            return True
        else:
            self.logger.warn(f"'{client_name}' is not registered in server.")
            return False

    def calculate(self) -> Any:
        return None

    def init_client_state(self) -> Any:
        return None

    def set_client_incremental_state(self, client_name: str, client_state: Dict) -> None:
        return None

    def set_client_integrated_state(self, client_name: str, client_state: Dict) -> None:
        return None

    def get_dispatch_incremental_state(self, client_name: str) -> Dict:
        return None

    def get_dispatch_integrated_state(self, client_name: str) -> Dict:
        return None
