import os
from typing import Dict, Callable, Any, List

import torch.nn as nn
from torch.optim import Optimizer

from criterions import criterions
from datasets.datasets_pipeline import ReIDTaskPipeline
from methods import methods
from models import nets, optimizers, schedulers
from modules.client import ClientModule
from modules.model import ModelModule
from modules.server import ServerModule


def parser_model(method_name: str, model_config: Dict) -> nn.Module:
    factory_kwargs = {n: p for n, p in model_config.items() if n not in ['name', 'fine_tuning']}
    net = nets[model_config['name']](**factory_kwargs)
    if model_config['fine_tuning']:
        for p in net.parameters():
            p.requires_grad = False
        for layer_name in model_config['fine_tuning']:
            for p in net.get_submodule(layer_name).parameters():
                p.requires_grad = True

    if hasattr(methods[method_name], 'Model'):
        return methods[method_name].Model(net=net, **factory_kwargs)
    else:
        return ModelModule(net)


def parser_criterion(criterion_configs: Any) -> List[Callable]:
    if isinstance(criterion_configs, dict):
        criterion_configs = [criterion_configs]

    loss_fn = []
    for criterion_config in criterion_configs:
        factory_kwargs = {n: p for n, p in criterion_config.items() if n not in ['name']}
        criterion = criterions[criterion_config['name']](**factory_kwargs)
        loss_fn.append(criterion)
    return loss_fn


def parser_optimizer(model: nn.Module, optim_config: Dict) -> Optimizer:
    factory_kwargs = {n: p for n, p in optim_config.items() if n not in ['name']}
    tuning_params = (p for p in model.net.parameters() if p.requires_grad)
    return optimizers[optim_config['name']](params=tuning_params, **factory_kwargs)


def parser_scheduler(optim: Optimizer, scheduler_config: Dict) -> Optimizer:
    factory_kwargs = {n: p for n, p in scheduler_config.items() if n not in ['name']}
    return schedulers[scheduler_config['name']](optimizer=optim, **factory_kwargs)


def parser_server(exp_config: Dict, common_config: Dict) -> ServerModule:
    model = parser_model(exp_config['exp_method'], exp_config['model_opts'])
    criterion = parser_criterion(exp_config['criterion_opts'])
    optimizer = parser_optimizer(model, exp_config['optimizer_opts'])
    scheduler = parser_scheduler(optimizer, exp_config['scheduler_opts'])
    operator = methods[exp_config['exp_method']].Operator(
        method_name=exp_config['exp_method'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    kwarg_factory = {n: p for n, p in exp_config['server'].items() if n not in ['server_name']}
    return methods[exp_config['exp_method']].Server(
        server_name=exp_config['server']['server_name'],
        model=model,
        operator=operator,
        ckpt_root=os.path.join(common_config['checkpoints_dir'], exp_config['exp_name']),
        **kwarg_factory
    )


def parser_clients(exp_config: Dict, common_config: Dict) -> ClientModule:
    clients = []
    for client_config in exp_config['clients']:
        model = parser_model(exp_config['exp_method'], exp_config['model_opts'])
        criterion = parser_criterion(exp_config['criterion_opts'])
        optimizer = parser_optimizer(model, exp_config['optimizer_opts'])
        scheduler = parser_scheduler(optimizer, exp_config['scheduler_opts'])
        operator = methods[exp_config['exp_method']].Operator(
            method_name=exp_config['exp_method'],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        task_pipeline = ReIDTaskPipeline(
            task_list=client_config['tasks'],
            task_opts=exp_config['task_opts'],
            datasets_dir=common_config['datasets_dir']
        )
        kwarg_factory = {n: p for n, p in client_config.items() if n not in ['client_name']}
        clients.append(methods[exp_config['exp_method']].Client(
            client_name=client_config['client_name'],
            model=model,
            operator=operator,
            ckpt_root=os.path.join(common_config['checkpoints_dir'], exp_config['exp_name']),
            task_pipeline=task_pipeline,
            **kwarg_factory
        ))

    return clients
