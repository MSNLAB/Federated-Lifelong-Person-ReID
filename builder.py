import os
from typing import Dict, Callable, Tuple, Any, List

import torch.nn as nn
from torch.optim import Optimizer

from criterions import get_callable_criterion
from datasets.datasets_pipeline import ReIDTaskPipeline
from methods import generate_server, generate_operator, generate_client, generate_model
from models import get_callable_model, get_callable_optimizer, get_callable_scheduler
from modules.client import ClientModule
from modules.server import ServerModule


def parser_model(method_name: str, model_config: Dict, device: str) -> nn.Module:
    net_constructor = get_callable_model(model_config['name'])
    factory_kwargs = model_config['arguments'] if 'arguments' in model_config.keys() else {}
    net = net_constructor(**factory_kwargs)
    return generate_model(method_name, net, device, **factory_kwargs)


def parser_criterion(criterion_configs: Any) -> List[Tuple[Callable, Dict]]:
    if isinstance(criterion_configs, dict):
        criterion_configs = [criterion_configs]
    criterions = []
    for criterion_config in criterion_configs:
        callable_criterion = get_callable_criterion(criterion_config['name'])
        factory_kwargs = criterion_config['arguments'] if 'arguments' in criterion_config.keys() else {}
        criterions.append(callable_criterion(**factory_kwargs))
    return criterions


def parser_optimizer(model: nn.Module, optim_config: Dict) -> Optimizer:
    optimizer = get_callable_optimizer(optim_config['name'])
    factory_kwargs = optim_config['arguments'] if 'arguments' in optim_config.keys() else {}

    if not optim_config['fine_tuning']:
        tuning_params = (p for p in model.parameters() if p.requires_grad)
    else:
        tuning_params = (p for layer_name in optim_config['fine_tuning_layers'] \
                         for p in model.get_submodule(layer_name).parameters())

    return optimizer(params=tuning_params, **factory_kwargs)


def parser_scheduler(optim: Optimizer, scheduler_config: Dict) -> Optimizer:
    scheduler = get_callable_scheduler(scheduler_config['name'])
    factory_kwargs = scheduler_config['arguments'] if 'arguments' in scheduler_config.keys() else {}
    return scheduler(optimizer=optim, **factory_kwargs)


def parser_server(
        job_name: str,
        method_name: str,
        server_config: Dict,
        common_config: Dict,
        **kwargs
) -> ServerModule:
    model = parser_model(method_name, server_config['model'], common_config['device'])
    criterion = parser_criterion(server_config['criterion'])
    optimizer = parser_optimizer(model, server_config['optimizer'])
    scheduler = parser_scheduler(optimizer, server_config['scheduler'])
    return generate_server(
        method_name=method_name,
        server_name=server_config['name'],
        model=model,
        operator=generate_operator(
            method_name=method_name,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=common_config['device']
        ),
        ckpt_root=os.path.join(common_config['checkpoints'], job_name),
        random_seed=common_config['random_seed'],
        **kwargs
    )


def parser_client(
        job_name: str,
        method_name: str,
        client_config: Dict,
        common_config: Dict,
        **kwargs
) -> ClientModule:
    model = parser_model(method_name, client_config['model'], common_config['device'])
    criterion = parser_criterion(client_config['criterion'])
    optimizer = parser_optimizer(model, client_config['optimizer'])
    scheduler = parser_scheduler(optimizer, client_config['scheduler'])

    task_list = []
    for task in client_config['tasks']:
        task_list.append({
            "task_name": task['task_name'],
            "augmentation": task['augmentation'],
            "epochs": task['epochs'],
            "batch_size": task['batch_size'],
            "sustained_round": task['sustained_round'],
            "img_size": task['img_size'],
            "norm_mean": task['norm_mean'],
            "norm_std": task["norm_std"],
            "dataset_paths":
                [os.path.join(common_config['datasets_base'], task['datasets'])] \
                    if isinstance(task['datasets'], str) else \
                    [os.path.join(common_config['datasets_base'], task_name) for task_name in task['datasets']]
        })

    model_ckpt_name = None
    if 'model_ckpt_name' in client_config.keys():
        model_ckpt_name = client_config['model_ckpt_name']

    task_pipeline = ReIDTaskPipeline(
        task_list,
        num_workers=client_config['workers'],
        pin_memory=client_config['pin_memory']
    )

    return generate_client(
        method_name=method_name,
        client_name=client_config['name'],
        model=model,
        operator=generate_operator(
            method_name=method_name,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=common_config['device']
        ),
        ckpt_root=os.path.join(common_config['checkpoints'], job_name),
        model_ckpt_name=model_ckpt_name,
        task_pipeline=task_pipeline,
        random_seed=common_config['random_seed'],
        **kwargs
    )
