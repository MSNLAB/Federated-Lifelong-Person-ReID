import os
import random
from typing import Dict, List, Tuple, Set, Union

import numpy as np
import torch
import torch.nn as nn


def torch_device(default_device: str = None, **kwargs) -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if default_device is not None:
        device = default_device if default_device in ['cuda', 'cpu'] else device
    else:
        device = kwargs['device'] if 'device' in kwargs.keys() else device
    return device


def tensor_reverse_permute(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None:
        return None
    dims = list(range(len(tensor.shape)))
    dims.reverse()
    return torch.permute(tensor, dims)


def extract_kwargs(kwargs: Dict, key: str, default_value=None):
    return kwargs[key] if key in kwargs else default_value


def params_state_size(params_state) -> int:
    if isinstance(params_state, (int, float, bool, complex, str)):
        return 1
    if isinstance(params_state, torch.Tensor):
        return params_state.numel()
    if isinstance(params_state, Dict):
        return sum([params_state_size(v) for k, v in params_state.items()])
    if isinstance(params_state, (List, Tuple, Set)):
        return sum([params_state_size(v) for v in params_state])
    raise Exception(f'unrecognized state type {type(params_state)} to calculate parameters size')


def extract_losses(losses) -> Union[torch.Tensor, float]:
    if isinstance(losses, torch.Tensor):
        return losses if losses.requires_grad and len(losses.shape) == 0 else 0.0
    if isinstance(losses, Dict):
        return sum([extract_losses(_loss) for _loss in losses.values()])
    if isinstance(losses, (List, Tuple, Set)):
        return sum([extract_losses(_loss) for _loss in losses])
    return 0.0


def random_shuffle(seed, _list):
    random.seed(seed)
    random.shuffle(_list)


def random_sample(seed, _list, num_pick):
    random.seed(seed)
    return random.sample(_list, num_pick)


def random_int(seed, start, end):
    random.seed(seed)
    random.randint(start, end)


def normalize(x, ord: int = None, axis: int = 0, keepdims=True):
    x_norm = np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    x = x / x_norm
    return x


def np_save(base_dir, filename, data):
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    np.save(os.path.join(base_dir, filename), data)


def load_task(base_dir, task):
    return np.load(os.path.join(base_dir, task), allow_pickle=True)


def same_seeds(seed=42069):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_value(*tensors: torch.Tensor):
    if len(tensors) == 1:
        return tensors[0].cpu().detach().item()
    else:
        return (tensor_value(tensor) for tensor in tensors)


class model_on_device(object):

    def __init__(self, model: nn.Module, device: str = 'cpu') -> None:
        self.model = model
        self.device = device
        super(model_on_device, self).__init__()

    def __enter__(self):
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        self.model.to(self.device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.cpu()
