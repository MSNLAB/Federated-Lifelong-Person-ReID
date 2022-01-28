from typing import Any, Optional, Union, Dict

import torch.nn as nn


class ModelModule(nn.Module):

    def __init__(self, net: nn.Module):
        super(ModelModule, self).__init__()
        self.net = net

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def cpu(self) -> Any:
        return super().cpu()

    def cuda(self, device: Optional[Union[int, Any]] = None) -> Any:
        return super().cuda()

    def to(self, device: str = 'cpu'):
        return super().to(device)

    def model_state(self, *args, **kwargs) -> Dict:
        raise NotImplementedError

    def update_model(self, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device
