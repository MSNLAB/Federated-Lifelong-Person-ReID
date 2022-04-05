import collections
import math
from queue import Queue
from typing import Any, Dict, Union, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn, fx
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from datasets.datasets_loader import ReIDImageDataset
from modules.client import ClientModule
from modules.model import ModelModule
from modules.operator import OperatorModule
from modules.server import ServerModule
from tools.distance import compute_euclidean_distance
from tools.evaluate import calculate_similarity_distance, evaluate
from tools.utils import model_on_device, ModulePathTracer


class AdaptiveLayer(nn.Module):

    def __init__(
            self,
            global_weight: torch.Tensor,
            global_weight_atten: torch.Tensor = None,
            adaptive_weight: torch.Tensor = None,
            adaptive_bias: torch.Tensor = None,
            atten_default: float = 0.80,
            **kwargs
    ):
        super(AdaptiveLayer, self).__init__()

        self.global_weight = Parameter()
        self.global_weight_atten = Parameter()
        self.adaptive_weight = Parameter()
        self.adaptive_bias = Parameter() if adaptive_bias is not None else None
        self.atten_default = atten_default

        self.initial_global_weight_atten = Parameter()
        self.initial_adaptive_weight = Parameter()

        self.init_training_weights(
            global_weight,
            global_weight_atten,
            adaptive_weight,
            adaptive_bias
        )

    def init_training_weights(
            self,
            global_weight=None,
            global_weight_atten=None,
            adaptive_weight=None,
            adaptive_bias=None,
    ) -> None:
        if global_weight is None:
            global_weight = self.global_weight.data
        self.global_weight.data = global_weight
        self.global_weight.requires_grad = False

        if global_weight_atten is None:
            global_weight_atten = torch.ones(self.global_weight.data.shape[-1]) * self.atten_default
        self.global_weight_atten.data = global_weight_atten
        self.initial_global_weight_atten.data = global_weight_atten.clone().detach()
        self.global_weight_atten.requires_grad = False

        if adaptive_weight is None:
            adaptive_weight = (1.0 - global_weight_atten.clone().detach()) \
                              * (global_weight.clone().detach())
        self.adaptive_weight.data = adaptive_weight
        self.initial_adaptive_weight.data = adaptive_weight.clone().detach()
        self.adaptive_weight.requires_grad = True

        if self.adaptive_bias is not None:
            if adaptive_bias is None:
                adaptive_bias = self.adaptive_bias.data
            self.adaptive_bias.data = adaptive_bias
            self.adaptive_bias.requires_grad = True

    def forward(self, data: torch.Tensor) -> Any:
        theta = self.global_weight_atten * self.global_weight + self.adaptive_weight
        bias = self.adaptive_bias

        return F.linear(
            input=data,
            weight=theta,
            bias=bias
        )


class AdaptiveConv2D(AdaptiveLayer):

    def __init__(
            self,
            global_weight: torch.Tensor,
            adaptive_weight: torch.Tensor = None,
            adaptive_bias: torch.Tensor = None,
            global_weight_atten: torch.Tensor = None,
            atten_default: float = 0.80,
            stride: int = 1,
            padding: int = 0,
            **kwargs
    ) -> None:
        super().__init__(
            global_weight,
            adaptive_weight,
            adaptive_bias,
            global_weight_atten,
            atten_default,
            **kwargs
        )
        self.stride = stride
        self.padding = padding

    def forward(self, data: torch.Tensor) -> Any:
        theta = self.global_weight_atten * self.global_weight + self.adaptive_weight
        bias = self.adaptive_bias

        return F.conv2d(
            input=data,
            weight=theta,
            bias=bias,
            stride=self.stride,
            padding=self.padding
        )


class AdaptiveBatchNorm(AdaptiveLayer):

    def __init__(
            self,
            global_weight: torch.Tensor,
            global_weight_atten: torch.Tensor = None,
            adaptive_weight: torch.Tensor = None,
            adaptive_bias: torch.Tensor = None,
            atten_default: float = 0.80,
            running_mean: torch.Tensor = None,
            running_var: torch.Tensor = None,
            num_batches_tracked: torch.Tensor = None,
            track_running_stats: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
            **kwargs
    ) -> None:
        super(AdaptiveBatchNorm, self).__init__(
            global_weight,
            adaptive_weight,
            adaptive_bias,
            global_weight_atten,
            atten_default,
            **kwargs
        )
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)
        self.register_buffer('num_batches_tracked', num_batches_tracked)
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.eps = eps

    def forward(self, data: torch.Tensor):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        theta = self.global_weight_atten * self.global_weight + self.adaptive_weight
        bias = self.adaptive_bias

        return F.batch_norm(
            input=data,
            running_mean=self.running_mean if not self.training or self.track_running_stats else None,
            running_var=self.running_var if not self.training or self.track_running_stats else None,
            weight=theta,
            bias=bias,
            training=bn_training,
            momentum=exponential_average_factor,
            eps=self.eps,
        )


class AdaptiveLayerNorm(AdaptiveLayer):

    def __init__(
            self,
            global_weight: torch.Tensor,
            global_weight_atten: torch.Tensor = None,
            adaptive_weight: torch.Tensor = None,
            adaptive_bias: torch.Tensor = None,
            atten_default: float = 0.80,
            normalized_shape: Tuple[int, ...] = None,
            eps: float = 1e-5,
            **kwargs
    ) -> None:
        super(AdaptiveLayerNorm, self).__init__(
            global_weight,
            adaptive_weight,
            adaptive_bias,
            global_weight_atten,
            atten_default,
            **kwargs
        )
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, data: torch.Tensor):
        theta = self.global_weight_atten * self.global_weight + self.adaptive_weight
        bias = self.adaptive_bias
        return F.layer_norm(data, self.normalized_shape, theta, bias, self.eps)


class Model(ModelModule):
    _module_transform_lut = {
        nn.Linear: AdaptiveLayer,
        nn.Conv2d: AdaptiveConv2D,
        # nn.BatchNorm2d: AdaptiveBatchNorm,
        # nn.LayerNorm: AdaptiveLayerNorm,
    }

    def __init__(
            self,
            net: Union[nn.Sequential, nn.Module],
            lambda_l1: float = 1e-4,
            lambda_k: int = 8000,
            atten_default: float = 0.80,
            **kwargs
    ) -> None:
        super(Model, self).__init__(net)

        self.atten_default = atten_default
        self.lambda_l1 = lambda_l1
        self.lambda_k = lambda_k
        self.args = kwargs

        self.layer_convert(self.net)

        self.ids = set()
        self.examplars = {}

        self.features_extractor = self.net.base

        # find training header
        tracer = ModulePathTracer()
        for node in tracer.trace(self.net).nodes:
            module_name = tracer.node_to_originating_module.get(node)
            if module_name:
                module = self.net.get_submodule(module_name)
                flag = False
                for sub_module in module.modules():
                    if isinstance(sub_module, AdaptiveLayer):
                        self.head_adaptive_layer = module
                        self.head_adaptive_layer.input_features = []
                        self.head_adaptive_layer.input_person_ids = []
                        self.head_adaptive_layer.input_classes = []
                        flag = True
                if flag:
                    break

        # generate graph module of training part
        tracer = ModulePathTracer()
        input_node = None
        for node in tracer.trace(self.net).nodes:
            if input_node is None:
                input_node = node
            module_name = tracer.node_to_originating_module.get(node)
            if module_name:
                if self.net.get_submodule(module_name) != self.head_adaptive_layer:
                    node.replace_all_uses_with(replace_with=input_node)
                    tracer.graph.erase_node(node)
                else:
                    self.training_graph = fx.GraphModule(self.net, tracer.graph)
                    break

    def layer_convert(self, net):
        leaves = self.module_leaves(net)
        for name, module in leaves:
            transform_flag = False
            if type(module) in self._module_transform_lut.keys():
                transform_flag = True
                for param in module.parameters():
                    if not param.requires_grad:
                        transform_flag = False
            if not transform_flag:
                continue

            if isinstance(module, nn.Linear):
                module = AdaptiveLayer(
                    global_weight=module.weight,
                    adaptive_bias=module.bias,
                    atten_default=self.atten_default,
                )

            if isinstance(module, nn.Conv2d):
                module = AdaptiveConv2D(
                    global_weight=module.weight,
                    adaptive_bias=module.bias,
                    atten_default=self.atten_default,
                    stride=module.stride,
                    padding=module.padding,
                )

            # if isinstance(module, nn.BatchNorm2d):
            #     module = AdaptiveBatchNorm(
            #         global_weight=module.weight,
            #         adaptive_bias=module.bias,
            #         atten_default=self.atten_default,
            #         running_mean=module.running_mean,
            #         running_var=module.running_var,
            #         num_batches_tracked=module.num_batches_tracked,
            #         track_running_stats=module.track_running_stats,
            #         momentum=module.momentum,
            #         eps=module.eps,
            #     )

            # if isinstance(module, nn.LayerNorm):
            #     module = AdaptiveLayerNorm(
            #         global_weight=module.weight,
            #         adaptive_bias=module.bias,
            #         atten_default=self.atten_default,
            #         normalized_shape=module.normalized_shape,
            #         eps=module.eps,
            #     )

            # replace module with adaptive layer
            pa_module = net
            name_path = name.split('.')
            for deep, module_name in enumerate(name_path, 1):
                if deep == len(name_path):
                    pa_module.add_module(module_name, module)
                else:
                    pa_module = pa_module.__getattr__(module_name)

    @property
    def m(self):
        return math.ceil(self.lambda_k / len(self.ids))

    def build_examplars(self, proto_loader, person_ids, device):
        protos, ids, classes, features = [], [], [], []
        self.eval()
        for data, person_id, classes_id in proto_loader:
            data = data.to(device)
            protos.append(data.clone().detach().cpu())
            ids.append(person_id.clone().detach().cpu())
            classes.append(classes_id.clone().detach().cpu())
            features.append(self.training_graph(data)[1].clone().detach().cpu())

        protos = torch.cat(protos).detach().numpy()
        ids = torch.cat(ids).detach().numpy()
        classes = torch.cat(classes).detach().numpy()
        features = torch.cat(features).detach().numpy()

        del_ids = []
        for idx, person_id in enumerate(ids):
            if len(person_ids) and person_id not in person_ids:
                del_ids.append(idx)

        protos = np.delete(protos, del_ids, axis=0)
        ids = np.delete(ids, del_ids, axis=0)
        classes = np.delete(classes, del_ids, axis=0)
        features = np.delete(features, del_ids, axis=0)

        for person_idx in np.unique(ids):
            _ids = np.argwhere(ids == person_idx).squeeze(axis=1)

            _protos = protos[_ids]
            _classes = classes[_ids]
            _features = features[_ids]
            _mean = sum(_features) / len(_features)

            examplars = []
            examplars_fea = []
            for i in range(self.m):
                p = _mean - (_features + np.sum(examplars_fea, axis=0)) / (i + 1)
                p = np.linalg.norm(p, axis=1)
                min_idx = np.argmin(p)
                examplars.append((_protos[min_idx], _classes[min_idx]))
                examplars_fea.append(_features[min_idx])

            self.examplars[person_idx] = examplars

    def reduce_examplars(self):
        for class_idx in self.examplars.keys():
            self.examplars[class_idx] = self.examplars[class_idx][:self.m]

    @staticmethod
    def module_leaves(model: nn.Module) -> List:
        leaves = []
        q = Queue()
        for name, module in model.named_children():
            q.put((name, module))
        while not q.empty():
            q_name, q_module = q.get()
            client_cnt = 0
            for name, module in q_module.named_children():
                q.put((f'{q_name}.{name}', module))
                client_cnt += 1
            if client_cnt == 0:
                leaves.append((q_name, q_module))
        return leaves

    def pre_trained_module_leaves(self, net: Optional[nn.Module] = None) -> List:
        if net is None:
            net = self.net
        return [(name, module) for name, module in self.module_leaves(net) \
                if not type(module) in self._module_transform_lut.values()]

    def adaptive_module_leaves(self, net: Optional[nn.Module] = None) -> List:
        if net is None:
            net = self.net
        return [(name, module) for name, module in self.module_leaves(net) \
                if type(module) in self._module_transform_lut.values()]

    def forward(self, data: torch.Tensor) -> Any:
        return self.net(data)

    def cpu(self) -> Any:
        self.net.cpu()
        return super().cpu()

    def cuda(self, device: Optional[Union[int, Any]] = None) -> Any:
        self.net.cuda()
        return super().cuda()

    def to(self, device: str = 'cpu'):
        self.net.to(device)
        return super().to(device)

    def model_state(self) -> Dict:
        adaptive_layers = self.adaptive_module_leaves()

        global_weights = {
            f'{name}.global_weight': layer.global_weight.clone().detach() \
            for name, layer in adaptive_layers \
            if layer.global_weight is not None
        }
        global_weight_atten = {
            f'{name}.global_weight_atten': layer.global_weight_atten.clone().detach() \
            for name, layer in adaptive_layers \
            if layer.global_weight_atten is not None
        }
        adaptive_weights = {
            f'{name}.adaptive_weight': layer.adaptive_weight.clone().detach() \
            for name, layer in adaptive_layers \
            if layer.adaptive_weight is not None
        }
        adaptive_bias = {
            f'{name}.adaptive_bias': layer.adaptive_bias.clone().detach() \
            for name, layer in adaptive_layers \
            if layer.adaptive_bias is not None
        }
        bn_params = {
            **{f'{name}.running_mean': layer.running_mean.clone().detach() \
               for name, layer in adaptive_layers \
               if isinstance(layer, AdaptiveBatchNorm) and layer.running_mean is not None},
            **{f'{name}.running_var': layer.running_var.clone().detach() \
               for name, layer in adaptive_layers \
               if isinstance(layer, AdaptiveBatchNorm) and layer.running_mean is not None},
            **{f'{name}.num_batches_tracked': layer.num_batches_tracked \
               for name, layer in adaptive_layers \
               if isinstance(layer, AdaptiveBatchNorm) and layer.running_mean is not None},
        }
        pre_trained_params = {
            f'{l_name}.{p_name}': params.clone().detach() \
            for l_name, layer in self.pre_trained_module_leaves() \
            for p_name, params in layer.state_dict().items()
        }

        return {
            'global_weight': global_weights,
            'global_weight_atten': global_weight_atten,
            'adaptive_weights': adaptive_weights,
            'adaptive_bias': adaptive_bias,
            'bn_params': bn_params,
            'pre_trained_params': pre_trained_params,
        }

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        global_weight = {}
        if 'global_weight' in params_state.keys():
            global_weight = {
                n: p.clone().detach() \
                for n, p in params_state['global_weight'].items()
            }

        global_weight_atten = {}
        if 'global_weight_atten' in params_state.keys():
            global_weight_atten = {
                n: p.clone().detach() \
                for n, p in params_state['global_weight_atten'].items()
            }

        adaptive_weights = {}
        if 'adaptive_weights' in params_state.keys():
            adaptive_weights = {
                n: p.clone().detach() \
                for n, p in params_state['adaptive_weights'].items()
            }

        adaptive_bias = {}
        if 'adaptive_bias' in params_state.keys():
            adaptive_bias = {
                n: p.clone().detach() \
                for n, p in params_state['adaptive_bias'].items()
            }

        bn_params = {}
        if 'bn_params' in params_state.keys():
            bn_params = {
                n: p.clone().detach() \
                for n, p in params_state['bn_params'].items()
            }

        pre_trained_params = {}
        if 'pre_trained_params' in params_state.keys():
            pre_trained_params = {
                n: p.clone().detach() \
                for n, p in params_state['pre_trained_params'].items()
            }

        model_params = {
            **global_weight,
            **global_weight_atten,
            **adaptive_weights,
            **adaptive_bias,
            **bn_params,
            **pre_trained_params,
        }

        model_dict = self.net.state_dict()
        for i, (n, p) in enumerate(model_params.items()):
            model_dict[n] = p.clone().detach()
        self.net.load_state_dict(model_dict)


class Operator(OperatorModule):

    def set_optimizer_parameters(self, model: Model):
        optimizer_param_factory = {n: p for n, p in self.optimizer.defaults.items()}
        params = [p for p in model.net.parameters() if p.requires_grad]
        self.optimizer.param_groups = [{'params': params, **optimizer_param_factory}]

    @staticmethod
    def generate_proto_loader(model, source_loader: DataLoader):
        def _task_token_hook(layer, fea_in, fea_out):
            layer.input_features.append(fea_in[0].cpu().detach().clone())

        model.head_adaptive_layer.input_features = []
        model.head_adaptive_layer.input_person_ids = []
        model.head_adaptive_layer.input_classes = []

        hook = model.head_adaptive_layer.register_forward_hook(_task_token_hook)

        model.eval()
        for data, person_id, classes_id in source_loader:
            model.head_adaptive_layer.input_person_ids.append(person_id)
            model.head_adaptive_layer.input_classes.append(classes_id)
            model.forward(data.to(model.device))

        model.head_adaptive_layer.input_features = torch.cat(model.head_adaptive_layer.input_features)
        model.head_adaptive_layer.input_person_ids = torch.cat(model.head_adaptive_layer.input_person_ids)
        model.head_adaptive_layer.input_classes = torch.cat(model.head_adaptive_layer.input_classes)

        protos = {}
        for prototype, person_id, classes_id in zip(
                model.head_adaptive_layer.input_features,
                model.head_adaptive_layer.input_person_ids,
                model.head_adaptive_layer.input_classes
        ):
            prototype, person_id, classes_id = prototype.numpy(), int(person_id), int(classes_id)
            if person_id not in protos.keys():
                protos[person_id] = []
            protos[person_id].append((prototype, classes_id))

        dataset = ConcatDataset([
            ReIDImageDataset(source=model.examplars),
            ReIDImageDataset(source=protos),
        ])

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=source_loader.batch_size,
            num_workers=source_loader.num_workers,
            pin_memory=source_loader.pin_memory,
            drop_last=len(dataset) % source_loader.batch_size == 1,
            persistent_workers=source_loader.persistent_workers,
            multiprocessing_context=source_loader.multiprocessing_context,
        )

        model.head_adaptive_layer.input_features = model.head_adaptive_layer.input_features.view(
            model.head_adaptive_layer.input_features.shape[0], -1)

        task_token = torch.mean(model.head_adaptive_layer.input_features, dim=0)

        model.head_adaptive_layer.input_features = []
        model.head_adaptive_layer.input_person_ids = []
        model.head_adaptive_layer.input_classes = []

        hook.remove()

        return dataloader, task_token

    def invoke_train(
            self,
            model: Model,
            dataloader: DataLoader,
            **kwargs
    ):
        train_acc = train_loss = 0.0
        batch_cnt = data_cnt = 0
        device = model.device
        dataloader, task_token = self.generate_proto_loader(model, dataloader)

        model.train()
        self.set_optimizer_parameters(model)
        for data, person_id, classes_id in dataloader:
            data, target = data.to(device), person_id.to(device)
            self.optimizer.zero_grad()
            stu_output = self._invoke_train(model.training_graph, data, target, **kwargs)
            score, feature, loss = stu_output['score'], stu_output['feature'], stu_output['loss']

            # l1 loss to make adaptive weight sparse
            adaptive_layers = model.adaptive_module_leaves()
            sparseness = 0.0
            for name, module in adaptive_layers:
                sparseness += torch.abs(module.initial_global_weight_atten - module.global_weight_atten).sum()
                sparseness += torch.abs(module.initial_adaptive_weight - module.adaptive_weight).sum()
            loss += sparseness * model.lambda_l1

            loss.backward()
            self.optimizer.step()
            train_acc += (torch.max(score, dim=1)[1] == target).sum().cpu().detach().item()
            train_loss += loss.cpu().detach().item()
            data_cnt += len(data)
            batch_cnt += 1

        train_acc = train_acc / data_cnt
        train_loss = train_loss / batch_cnt

        if self.scheduler:
            self.scheduler.step()

        return {
            'task_token': task_token,
            'proto_loader': dataloader,
            'accuracy': train_acc,
            'loss': train_loss,
            'batch_count': batch_cnt,
            'data_count': data_cnt,
        }

    def _invoke_train(
            self,
            model: Model,
            data: Any,
            target: Any,
            **kwargs
    ) -> Any:
        score, feature = model.forward(data)
        loss = 0.0
        for loss_func in self.criterion:
            loss += loss_func(score=score, feature=feature, target=target)

        return {
            'score': score,
            'feature': feature,
            'loss': loss,
        }

    def invoke_predict(
            self,
            model: Model,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        pred_acc = pred_loss = 0.0
        batch_cnt = data_cnt = 0
        device = model.device

        model.train()
        for data, person_id, classes_id in dataloader:
            data, target = data.to(device), person_id.to(device)
            with torch.no_grad():
                output = self._invoke_predict(model, data, target, **kwargs)
            score, loss = output['score'], output['loss']
            pred_acc += (torch.max(score, dim=1)[1] == target).sum().cpu().detach().item()
            pred_loss += loss.cpu().detach().item()
            data_cnt += len(data)
            batch_cnt += 1

        pred_acc = pred_acc / data_cnt
        pred_loss = pred_loss / batch_cnt

        return {
            'accuracy': pred_acc,
            'loss': pred_loss,
            'batch_count': batch_cnt,
            'data_count': data_cnt,
        }

    def _invoke_predict(
            self,
            model: Model,
            data: Any,
            target: Any,
            **kwargs
    ) -> Any:
        score, feature = model.forward(data)
        loss = 0.0
        for loss_func in self.criterion:
            loss += loss_func(score=score, feature=feature, target=target)

        return {
            'score': score,
            'feature': feature,
            'loss': loss
        }

    def invoke_inference(
            self,
            model: Model,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        batch_cnt, data_cnt = 0, 0
        device = model.device
        features = []

        model.eval()
        for data, person_id, classes_id in dataloader:
            data = data.to(device)
            with torch.no_grad():
                feature = self._invoke_inference(model, data, **kwargs)['feature']
                features.append(feature.clone().detach())
            data_cnt += len(data)
            batch_cnt += 1
        features = torch.cat(features, dim=0).cpu().detach()

        return {
            'features': features,
            'batch_count': batch_cnt,
            'data_count': data_cnt,
        }

    def _invoke_inference(
            self,
            model: Model,
            data: Any,
            norm: bool = True,
            **kwargs
    ) -> Any:
        feat = model.forward(data)
        if norm:
            feat = F.normalize(feat, dim=1, p=2)
        return {'feature': feat}

    def invoke_valid(
            self,
            model: Model,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        batch_cnt, data_cnt = 0, 0
        device = model.device
        features, labels = [], []

        model.eval()
        for data, person_id, classes_id in dataloader:
            data, target = data.to(device), person_id.to(device)
            with torch.no_grad():
                feature = self._invoke_valid(model, data, target)['feature']
                features.append(feature.clone().detach())
                labels.append(person_id.clone().detach())
            batch_cnt += 1
            data_cnt += len(data)

        features = torch.cat(features, dim=0).cpu().detach()
        labels = torch.cat(labels, dim=0).cpu().detach()

        return {
            'features': features,
            'labels': labels,
            'batch_count': batch_cnt,
            'data_count': data_cnt,
        }

    def _invoke_valid(
            self,
            model: Model,
            data: Any,
            target: Any,
            norm: bool = True,
            **kwargs
    ) -> Any:
        feat = model.forward(data)
        if norm:
            feat = F.normalize(feat, dim=1, p=2)
        return {'feature': feat}


class Client(ClientModule):

    def __init__(
            self,
            client_name: str,
            model: Model,
            operator: Operator,
            ckpt_root: str,
            model_ckpt_name: str = None,
            **kwargs
    ) -> None:
        super().__init__(client_name, model, operator, ckpt_root, model_ckpt_name, **kwargs)
        self.current_task = None
        self.task_token = None
        self.train_cnt = 0
        self.test_cnt = 0

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        self.model.update_model(params_state)

    def load_model(self, model_name: str):
        model_dict = self.model.model_state()
        model_dict = self.load_state(model_name, model_dict)
        self.model.update_model(model_dict)
        self.model.examplars = self.load_state(f'{model_name}_examplars', {})

    def save_model(self, model_name: str):
        model_dict = self.model.model_state()
        self.save_state(model_name, model_dict, True)
        self.save_state(f'{model_name}_examplars', self.model.examplars, True)

    def get_incremental_state(self, **kwargs) -> Dict:
        adaptive_layers = self.model.adaptive_module_leaves()
        model_state = self.model.model_state()
        incremental_shared_weights = {
            f'{name}.global_weight': (layer.global_weight_atten * layer.global_weight
                                      + layer.adaptive_weight).clone().detach() \
            for name, layer in adaptive_layers
        }
        return {
            'train_cnt': self.train_cnt,
            'task_token': self.task_token,
            'incremental_sw': incremental_shared_weights,
            'incremental_bn': model_state['bn_params'],
        }

    def get_integrated_state(self, **kwargs) -> Dict:
        adaptive_layers = self.model.adaptive_module_leaves()
        model_state = self.model.model_state()
        integrated_shared_weights = {
            f'{name}.global_weight': (layer.global_weight_atten * layer.global_weight
                                      + layer.adaptive_weight).clone().detach() \
            for name, layer in adaptive_layers
        }
        return {
            'train_cnt': self.train_cnt,
            'task_token': self.task_token,
            'integrated_sw': integrated_shared_weights,
            'integrated_bn': model_state['bn_params'],
            'pre_trained_params': model_state['pre_trained_params'],
        }

    def update_by_incremental_state(self, state: Dict, **kwargs) -> Any:
        model_params = {'global_weight': state['incremental_shared_params']}

        if self.current_task:
            if self.model_ckpt_name:
                self.load_model(self.model_ckpt_name)
            else:
                self.load_model(self.current_task)

        self.update_model(model_params)
        for _, adaptive_module in self.model.adaptive_module_leaves():
            adaptive_module.init_training_weights()

        self.logger.info('Update model succeed by incremental state from server.')

    def update_by_integrated_state(self, state: Dict, **kwargs) -> Any:
        model_params = {
            'global_weight': state['integrated_global_weight'],
            'bn_params': state['integrated_bn_params'],
            'pre_trained_params': state['integrated_pre_trained_params'],
        }

        if self.current_task:
            if self.model_ckpt_name:
                self.load_model(self.model_ckpt_name)
            else:
                self.load_model(self.current_task)

        self.update_model(model_params)
        for _, adaptive_module in self.model.adaptive_module_leaves():
            adaptive_module.init_training_weights()

        self.logger.info('Update model succeed by integrated state from server.')

    def train(
            self,
            epochs: int,
            task_name: str,
            tr_loader: DataLoader,
            val_loader: DataLoader,
            early_stop_threshold: int = 3,
            device: str = 'cpu',
            **kwargs
    ) -> Any:
        if self.current_task is None or self.current_task != task_name:
            self.model.ids.update(tr_loader.dataset.person_ids)
        self.current_task = task_name

        output = {}
        perf_loss, perf_acc, sustained_cnt = 1e8, 0, 0
        initial_lr = self.operator.optimizer.defaults['lr']
        task_tokens = []

        with model_on_device(self.model, device):
            for epoch in range(1, epochs + 1):
                output = self.train_one_epoch(task_name, tr_loader, val_loader)
                task_token, data_count = output['task_token'], output['data_count']
                accuracy, loss = output['accuracy'], output['loss']

                # early stop
                sustained_cnt += 1
                if loss <= perf_loss and accuracy >= perf_acc:
                    perf_loss, perf_acc = loss, accuracy
                    sustained_cnt = 0
                if early_stop_threshold and sustained_cnt >= early_stop_threshold:
                    break

                task_tokens.append(task_token)
                self.train_cnt += data_count

                self.logger.info_train(
                    task_name, device,
                    data_count, perf_acc, perf_loss,
                    epoch, epochs
                )

            self.model.reduce_examplars()
            self.model.build_examplars(output['proto_loader'], tr_loader.dataset.person_ids, device)

        # Reset learning rate
        self.operator.optimizer.state = collections.defaultdict(dict)
        for param_group in self.operator.optimizer.param_groups:
            param_group['lr'] = initial_lr

        self.task_token = sum(task_tokens) / len(task_tokens)

        if self.model_ckpt_name:
            self.save_model(self.model_ckpt_name)
        else:
            self.save_model(self.current_task)
        return output

    def train_one_epoch(
            self,
            task_name: str,
            tr_loader: DataLoader,
            val_loader: DataLoader,
            **kwargs
    ) -> Any:
        return self.operator.invoke_train(self.model, tr_loader)

    def inference(
            self,
            task_name: str,
            query_loader: Union[List[DataLoader], DataLoader],
            gallery_loader: Union[List[DataLoader], DataLoader],
            device: str = 'cpu',
            **kwargs
    ) -> Any:
        if self.model_ckpt_name:
            self.load_model(self.model_ckpt_name)
        else:
            self.load_model(task_name)

        with model_on_device(self.model, device):
            gallery_features = self.operator.invoke_inference(self.model, gallery_loader)['features']
            query_features = self.operator.invoke_inference(self.model, query_loader)['features']

        self.test_cnt += len(gallery_features) + len(query_features)

        output = {query_id: {} for query_id in range(len(query_features))}
        for query_id in range(len(query_features)):
            similarity_distance = calculate_similarity_distance(
                query_feature=query_features[query_id],
                gallery_features=gallery_features
            )
            for gallery_id, distance in enumerate(similarity_distance):
                output[query_id][gallery_id] = distance

        return output

    def validate(
            self,
            task_name: str,
            query_loader: Union[List[DataLoader], DataLoader],
            gallery_loader: Union[List[DataLoader], DataLoader],
            device: str = 'cpu',
            **kwargs
    ) -> Any:
        if self.model_ckpt_name:
            self.load_model(self.model_ckpt_name)
        else:
            self.load_model(task_name)

        with model_on_device(self.model, device):
            gallery_output = self.operator.invoke_valid(self.model, gallery_loader)
            query_output = self.operator.invoke_valid(self.model, query_loader)

        gallery_size = len(gallery_output['features'])
        query_size = len(query_output['features'])

        self.test_cnt += gallery_size + query_size

        cmc, mAP = evaluate(
            query_features=query_output['features'],
            query_labels=query_output['labels'],
            gallery_features=gallery_output['features'],
            gallery_labels=gallery_output['labels'],
            device=device
        )

        avg_rep = torch.cat([query_output['features'], gallery_output['features']], dim=0)
        avg_rep = torch.sum(avg_rep, dim=0) / len(avg_rep)

        self.logger.info_validation(task_name, query_size, gallery_size, cmc, mAP)
        return cmc, mAP, avg_rep


class Server(ServerModule):
    def __init__(
            self,
            server_name: str,
            model: Model,
            operator: OperatorModule,
            ckpt_root: str,
            distance_calculate_step: int = 10,
            distance_calculate_decay: int = 0.8,
            **kwargs
    ):
        super().__init__(server_name, model, operator, ckpt_root, **kwargs)
        self.token_memory = {}
        self.distance_calculate_step = distance_calculate_step
        self.distance_calculate_decay = distance_calculate_decay

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        self.model.update_model(params_state)

    def load_model(self, model_name: str):
        model_dict = self.model.model_state()
        model_dict = self.load_state(model_name, model_dict)
        self.model.update_model(model_dict)

    def save_model(self, model_name: str):
        model_dict = self.model.model_state()
        self.save_state(model_name, model_dict, True)

    def calculate(self) -> Any:
        merge_incremental_params = {}
        train_total_cnt = sum([client['train_cnt'] for _, client in self.clients.items()])

        for cname, cstate in self.clients.items():
            # k, params = cstate['train_cnt'], {**cstate['incremental_sw'], **cstate['incremental_bn']}
            k, params = cstate['train_cnt'], cstate['incremental_sw']
            params = {
                n: (p.clone().detach() * k / train_total_cnt).type(dtype=p.dtype) \
                for n, p in params.items()
            }
            for n, p in params.items():
                if n not in merge_incremental_params.keys():
                    merge_incremental_params[n] = torch.zeros_like(p)
                merge_incremental_params[n] += p.clone().detach()

        model_dict = self.model.net.state_dict()
        for i, (n, p) in enumerate(merge_incremental_params.items()):
            model_dict[n] = p.clone().detach()
        self.model.net.load_state_dict(model_dict)

        self.save_state(f'{self.server_name}_tokens', self.token_memory, True)

    def set_client_incremental_state(self, client_name: str, client_state: Dict) -> None:
        if client_name not in self.clients.keys():
            self.logger.warn(f'Collect incremental state failed from unregistered client {client_name}.')
        else:
            self.clients[client_name] = client_state
            if client_name not in self.token_memory.keys():
                self.token_memory[client_name] = []
            self.token_memory[client_name].append(client_state['task_token'])
            self.logger.info(f'Collect incremental state successfully from client {client_name}.')

    def set_client_integrated_state(self, client_name: str, client_state: Dict) -> None:
        if client_name not in self.clients.keys():
            self.logger.warn(f'Collect integrated state failed from unregistered client {client_name}.')
        else:
            self.clients[client_name] = client_state
            if client_name not in self.token_memory.keys():
                self.token_memory[client_name] = []
            self.token_memory[client_name].append(client_state['task_token'])
            self.logger.info(f'Collect integrated state successfully from client {client_name}.')

    def get_dispatch_incremental_state(self, client_name: str) -> Dict:
        task_token = self.clients[client_name]['task_token'].unsqueeze(dim=0)
        select_client, token_distance = [], []

        for c_name, c_tokens in self.token_memory.items():
            c_tokens = c_tokens[::-1 * self.distance_calculate_step]  # search token in reverse order
            if c_name != client_name:
                dis = 1e-8
                for decay_cnt, other_token in enumerate(c_tokens):
                    other_token = other_token.unsqueeze(dim=0)
                    _dis = compute_euclidean_distance(task_token, other_token)  # euclidean distance between tasks
                    dis += _dis / math.pow(self.distance_calculate_decay, decay_cnt)  # weaken far tasks distance
                select_client.append(c_name)
                token_distance.append(1.0 / dis)  # correlation is the inverse of distance

        select_client.append(client_name)
        token_distance.append(sum(token_distance) / len(token_distance))

        token_distance = torch.Tensor(token_distance)
        token_distance = torch.nn.functional.softmax(token_distance, dim=0).tolist()

        merge_incremental_params = {}
        for c_name, dis in zip(select_client, token_distance):
            self.logger.info(f'Relevant ratio between {client_name} and {c_name}: {dis:.4f}')
            client_state = self.clients[c_name]
            # params = {**client_state['incremental_sw'], **client_state['incremental_bn']}
            params = client_state['incremental_sw']

            params = {
                n: (p.clone().detach() * dis).type(dtype=p.dtype) \
                for n, p in params.items()
            }
            for n, p in params.items():
                if n not in merge_incremental_params.keys():
                    merge_incremental_params[n] = torch.zeros_like(p)
                merge_incremental_params[n] += p.clone().detach()

        return {
            'incremental_shared_params': merge_incremental_params,
        }

    def get_dispatch_integrated_state(self, client_name: str) -> Dict:
        model_state = self.model.model_state()
        return {
            'integrated_global_weight': model_state['global_weight'],
            'integrated_bn_params': model_state['bn_params'],
            'integrated_pre_trained_params': model_state['pre_trained_params'],
        }
