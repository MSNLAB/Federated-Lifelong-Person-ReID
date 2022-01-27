import gc
import os
import random
from typing import Dict, List, Tuple, Set, Union, Any, Optional, Callable

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn


def torch_device(default_device: str = None, **kwargs) -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if default_device is not None:
        device = default_device if default_device in ['cuda', 'cpu'] else device
    else:
        device = kwargs['device'] if 'device' in kwargs.keys() else device
    return device


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(target.device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


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


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class model_on_device(object):

    def __init__(self, model: nn.Module, device: str = 'cpu') -> None:
        self.model = model
        self.device = device
        super(model_on_device, self).__init__()

    def __enter__(self):
        self.model.to(self.device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.cpu()


class ModulePathTracer(torch.fx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """

    # The current qualified name of the Module being traced. The top-level
    # module is signified by empty string. This is updated when entering
    # call_module and restored when exiting call_module
    current_module_qualified_name: str = ''
    # A map from FX Node to the qualname of the Module from which it
    # originated. This is recorded by `create_proxy` when recording an
    # operation
    node_to_originating_module: Dict[torch.fx.Node, str] = {}

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(m)
            return super().call_module(m, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        return proxy
