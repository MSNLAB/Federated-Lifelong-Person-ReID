from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from methods import baseline, ewc, mas, fedavg, fedprox, fedcurv, fedweit, ours
from methods.baseline import *
from modules.client import ClientModule
from modules.operator import OperatorModule

method_list = {
    'baseline': baseline,
    'ewc': ewc,
    'mas': mas,
    'fedavg': fedavg,
    'fedprox': fedprox,
    'fedcurv': fedcurv,
    'fedweit': fedweit,
    'ours': ours,
}


def get_method_constructor(method_name: str) -> Any:
    if method_name not in method_list.keys():
        raise ValueError(f"Could not find the method named '{method_name}'.")
    return method_list[method_name]


def generate_operator(
        method_name: str,
        criterion: _Loss,
        optimizer: Optimizer,
        scheduler: _LRScheduler = None,
        device: str = None
) -> OperatorModule:
    method = get_method_constructor(method_name)
    if hasattr(method, 'Operator'):
        return method.Operator(
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device
        )
    return None


def generate_model(
        method_name: str,
        net: Union[nn.Sequential, nn.Module],
        device: str = None,
        **kwargs
) -> nn.Module:
    method = get_method_constructor(method_name)
    if hasattr(method, 'Model'):
        return method.Model(net=net, device=device, **kwargs)
    return net


def generate_client(
        method_name: str,
        client_name: str,
        model: Union[nn.Sequential, nn.Module],
        operator: OperatorModule,
        ckpt_root: str,
        model_ckpt_name: str = None,
        **kwargs
) -> ClientModule:
    method = get_method_constructor(method_name)
    if hasattr(method, 'Client'):
        return method.Client(
            client_name=client_name,
            model=model,
            operator=operator,
            ckpt_root=ckpt_root,
            model_ckpt_name=model_ckpt_name,
            **kwargs
        )
    return None


def generate_server(
        method_name: str,
        server_name: str,
        model: Union[nn.Sequential, nn.Module],
        operator: OperatorModule,
        ckpt_root: str,
        **kwargs
) -> ClientModule:
    method = get_method_constructor(method_name)
    if hasattr(method, 'Server'):
        return method.Server(
            server_name=server_name,
            model=model,
            operator=operator,
            ckpt_root=ckpt_root,
            **kwargs
        )
    return None
