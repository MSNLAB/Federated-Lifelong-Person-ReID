###################################################################################################
# Code from: https://github.com/wyjeong/FedWeIT
# @inproceedings{yoon2021federated,
#   title={Federated continual learning with weighted inter-client transfer},
#   author={Yoon, Jaehong and Jeong, Wonyong and Lee, Giwoong and Yang, Eunho and Hwang, Sung Ju},
#   booktitle={International Conference on Machine Learning},
#   pages={12073--12086},
#   year={2021},
#   organization={PMLR}
# }
###################################################################################################

import collections
import copy
import random
from queue import Queue
from typing import Any, Dict, Union, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import Parameter
from torch.utils.data import DataLoader

from modules.client import ClientModule
from modules.model import ModelModule
from modules.operator import OperatorModule
from modules.server import ServerModule
from tools.evaluate import calculate_similarity_distance, evaluate
from tools.utils import tensor_reverse_permute, model_on_device


class DecomposedLayer(nn.Module):

    def __init__(
            self,
            shared_weight: torch.Tensor,
            bias: torch.Tensor = None,
            mask: torch.Tensor = None,
            adaptive: torch.Tensor = None,
            knowledge_base: torch.Tensor = None,
            atten: torch.Tensor = None,
            lambda_l1: float = None,
            lambda_mask: float = None,
            kb_cnt: int = 5,
            **kwargs
    ):
        super(DecomposedLayer, self).__init__()

        self.sw = Parameter()
        self.bias = Parameter() if bias is not None else None
        self.mask = Parameter()
        self.aw = Parameter()
        self.aw_kb = Parameter()
        self.atten = Parameter()

        if kb_cnt is None:
            kb_cnt = 5
        self.kb_cnt = kb_cnt

        if lambda_l1 is None:
            lambda_l1 = 1e-3
        self.lambda_l1 = lambda_l1

        if lambda_mask is None:
            lambda_mask = 0.0
        self.lambda_mask = lambda_mask

        self.init_training_weights(
            shared_weight,
            bias,
            mask,
            adaptive,
            knowledge_base,
            atten
        )

    def init_training_weights(
            self,
            shared_weight=None,
            bias=None,
            mask=None,
            adaptive=None,
            knowledge_base=None,
            atten=None,
    ):
        if shared_weight is None:
            shared_weight = tensor_reverse_permute(self.sw.data)
        self.sw.data = tensor_reverse_permute(shared_weight)
        self.sw.requires_grad = False

        self.bias = None
        if bias is not None:
            if self.bias is None:
                self.bias = Parameter()
            self.bias.data = tensor_reverse_permute(bias)
            self.bias.requires_grad = True

        if mask is None:
            mask = torch.sigmoid(torch.zeros(self.sw.shape[-1]))
        self.mask.data = mask
        self.mask.requires_grad = True

        if adaptive is None:
            adaptive = self.sw.data.clone().detach()
            adaptive = (1 - mask.data) * adaptive
        self.aw.data = adaptive
        self.aw.requires_grad = True

        if knowledge_base is None:
            knowledge_base = torch.zeros(
                size=[shape for shape in self.sw.shape] + [self.kb_cnt]
            )
        self.aw_kb.data = knowledge_base
        self.aw_kb.requires_grad = False

        if atten is None:
            atten = torch.zeros([self.kb_cnt])
        self.atten.data = atten
        self.atten.requires_grad = True

    @staticmethod
    def l1_pruning(weights: torch.Tensor, hyper_parameters: torch.Tensor):
        hard_threshold = torch.greater(torch.abs(weights), hyper_parameters).float()
        return weights * hard_threshold

    def forward(self, data: torch.Tensor) -> Any:
        aw = self.aw  # if not self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask  # if not self.training else self.l1_pruning(self.mask, self.lambda_mask)
        theta = mask * self.sw + aw + torch.sum(self.atten * self.aw_kb, dim=-1, keepdim=False)
        bias = self.bias
        return F.linear(
            input=data,
            weight=tensor_reverse_permute(theta),
            bias=tensor_reverse_permute(bias)
        )


class DecomposedConv2D(DecomposedLayer):

    def __init__(
            self,
            shared_weight: torch.Tensor,
            bias: torch.Tensor = None,
            stride: int = 1,
            padding: int = 0,
            mask: torch.Tensor = None,
            adaptive: torch.Tensor = None,
            knowledge_base: torch.Tensor = None,
            atten: torch.Tensor = None,
            lambda_l1: torch.Tensor = None,
            lambda_mask: torch.Tensor = None,
            kb_cnt: int = 5,
            **kwargs
    ):
        super(DecomposedConv2D, self).__init__(shared_weight, bias, mask, adaptive, knowledge_base,
                                               atten, lambda_l1, lambda_mask, kb_cnt, **kwargs)
        self.stride = stride
        self.padding = padding

    def forward(self, data: torch.Tensor) -> Any:
        aw = self.aw  # if not self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask  # if not self.training else self.l1_pruning(self.mask, self.lambda_mask)
        theta = mask * self.sw + aw + torch.sum(self.atten * self.aw_kb, dim=-1, keepdim=False)
        bias = self.bias
        return F.conv2d(
            input=data,
            weight=tensor_reverse_permute(theta),
            bias=tensor_reverse_permute(bias),
            stride=self.stride,
            padding=self.padding
        )


class DecomposedBatchNorm(DecomposedLayer):
    def __init__(
            self,
            shared_weight: torch.Tensor,
            bias: torch.Tensor = None,
            mask: torch.Tensor = None,
            adaptive: torch.Tensor = None,
            knowledge_base: torch.Tensor = None,
            atten: torch.Tensor = None,
            lambda_l1: float = None,
            lambda_mask: float = None,
            kb_cnt: int = 5,
            running_mean: torch.Tensor = None,
            running_var: torch.Tensor = None,
            num_batches_tracked: torch.Tensor = None,
            track_running_stats: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
            **kwargs
    ) -> None:
        super(DecomposedBatchNorm, self).__init__(shared_weight, bias, mask, adaptive, knowledge_base,
                                                  atten, lambda_l1, lambda_mask, kb_cnt, **kwargs)
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

        aw = self.aw  # if not self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask  # if not self.training else self.l1_pruning(self.mask, self.lambda_mask)
        theta = mask * self.sw + aw + torch.sum(self.atten * self.aw_kb, dim=-1, keepdim=False)
        bias = self.bias

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


class DecomposedLayerNorm(DecomposedLayer):

    def __init__(
            self,
            shared_weight: torch.Tensor,
            bias: torch.Tensor = None,
            mask: torch.Tensor = None,
            adaptive: torch.Tensor = None,
            knowledge_base: torch.Tensor = None,
            atten: torch.Tensor = None,
            lambda_l1: float = None,
            lambda_mask: float = None,
            kb_cnt: int = 5,
            normalized_shape: Tuple[int, ...] = None,
            eps: float = 1e-5,
            **kwargs
    ) -> None:
        super(DecomposedLayerNorm, self).__init__(shared_weight, bias, mask, adaptive, knowledge_base,
                                                  atten, lambda_l1, lambda_mask, kb_cnt, **kwargs)
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, data: torch.Tensor):
        aw = self.aw  # if not self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask  # if not self.training else self.l1_pruning(self.mask, self.lambda_mask)
        theta = mask * self.sw + aw + torch.sum(self.atten * self.aw_kb, dim=-1, keepdim=False)
        bias = self.bias
        return F.layer_norm(data, self.normalized_shape, theta, bias, self.eps)


class Model(ModelModule):
    _module_transform_lut = {
        nn.Linear: DecomposedLayer,
        nn.Conv2d: DecomposedConv2D,
        nn.BatchNorm2d: DecomposedBatchNorm,
        nn.LayerNorm: DecomposedLayerNorm,
    }

    def __init__(
            self,
            net: Union[nn.Sequential, nn.Module],
            lambda_l1: float = 1e-3,
            lambda_l2: float = 1e2,
            lambda_mask: float = 0.0,
            kb_cnt: int = 5,
            **kwargs
    ) -> None:
        super(Model, self).__init__(net)

        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_mask = lambda_mask
        self.kb_cnt = kb_cnt
        self.args = kwargs
        self.net_list = {}
        self.layer_convert(self.net)

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
                module = DecomposedLayer(
                    shared_weight=module.weight,
                    bias=module.bias,
                    lambda_l1=self.lambda_l1,
                    lambda_mask=self.lambda_mask,
                    kb_cnt=self.kb_cnt
                )

            if isinstance(module, nn.Conv2d):
                module = DecomposedConv2D(
                    shared_weight=module.weight,
                    bias=module.bias,
                    stride=module.stride,
                    padding=module.padding,
                    lambda_l1=self.lambda_l1,
                    lambda_mask=self.lambda_mask,
                    kb_cnt=self.kb_cnt
                )

            # if isinstance(module, nn.BatchNorm2d):
            #     module = DecomposedBatchNorm(
            #         shared_weight=module.weight,
            #         bias=module.bias,
            #         lambda_l1=self.lambda_l1,
            #         lambda_mask=self.lambda_mask,
            #         kb_cnt=self.kb_cnt,
            #         running_mean=module.running_mean,
            #         running_var=module.running_var,
            #         num_batches_tracked=module.num_batches_tracked,
            #         track_running_stats=module.track_running_stats,
            #         momentum=module.momentum,
            #         eps=module.eps,
            #     )
            #
            # if isinstance(module, nn.LayerNorm):
            #     module = DecomposedBatchNorm(
            #         shared_weight=module.weight,
            #         bias=module.bias,
            #         lambda_l1=self.lambda_l1,
            #         lambda_mask=self.lambda_mask,
            #         kb_cnt=self.kb_cnt,
            #         normalized_shape=module.normalized_shape,
            #         eps=module.eps,
            #     )

            # replace module with decomposed layer
            pa_module = net
            name_path = name.split('.')
            for i, module_name in enumerate(name_path):
                if i + 1 == len(name_path):
                    pa_module.add_module(module_name, module)
                else:
                    pa_module = pa_module.__getattr__(module_name)

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

    def pre_trained_module_leaves(self) -> List:
        return [(name, module) for name, module in self.module_leaves(self.net) \
                if not type(module) in self._module_transform_lut.values()]

    def decomposed_module_leaves(self) -> List:
        return [(name, module) for name, module in self.module_leaves(self.net) \
                if type(module) in self._module_transform_lut.values()]

    def remember_params(self, model_name: str):
        copied_net = copy.deepcopy(self.net)
        pre_trained_layers = self.pre_trained_module_leaves()
        for name, layers in pre_trained_layers:
            copied_net.__setattr__(name, layers)
        self.net_list[model_name] = copied_net

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
        decomposed_layers = self.decomposed_module_leaves()
        pre_trained_layers = self.pre_trained_module_leaves()

        shared_weights = {
            f'{name}.sw': layer.sw.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.sw is not None
        }
        adaptive_weights = {
            f'{name}.aw': layer.aw.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.aw is not None
        }
        mask_weights = {
            f'{name}.mask': layer.mask.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.mask is not None
        }
        bias_weights = {
            f'{name}.bias': layer.bias.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.bias is not None
        }
        atten_weights = {
            f'{name}.atten': layer.atten.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.atten is not None
        }
        kb_weights = {
            f'{name}.aw_kb': layer.aw_kb.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.aw_kb is not None
        }
        bn_params = {
            **{f'{name}.running_mean': layer.running_mean.clone().detach() \
               for name, layer in decomposed_layers \
               if isinstance(layer, DecomposedBatchNorm) and layer.running_mean is not None},
            **{f'{name}.running_var': layer.running_var.clone().detach() \
               for name, layer in decomposed_layers \
               if isinstance(layer, DecomposedBatchNorm) and layer.running_var is not None},
            **{f'{name}.num_batches_tracked': layer.num_batches_tracked \
               for name, layer in decomposed_layers \
               if isinstance(layer, DecomposedBatchNorm) and layer.num_batches_tracked is not None},
        }
        pre_trained_params = {
            f'{l_name}.{p_name}': params.clone().detach()
            for l_name, layer in pre_trained_layers \
            for p_name, params in layer.state_dict().items()
        }

        return {
            'sw': shared_weights,
            'aw': adaptive_weights,
            'mask': mask_weights,
            'bias': bias_weights,
            'atten': atten_weights,
            'aw_kb': kb_weights,
            'bn_params': bn_params,
            'pre_trained_params': pre_trained_params,
        }

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        if params_state and len(params_state) != 0:
            shared_weights = {}
            if 'sw' in params_state.keys():
                shared_weights = {
                    n: p.clone().detach() \
                    for n, p in params_state['sw'].items()
                }

            adaptive_weights = {}
            if 'aw' in params_state.keys():
                adaptive_weights = {
                    n: p.clone().detach() \
                    for n, p in params_state['aw'].items()
                }

            mask_weights = {}
            if 'mask' in params_state.keys():
                mask_weights = {
                    n: p.clone().detach() \
                    for n, p in params_state['mask'].items()
                }

            bias_weights = {}
            if 'bias' in params_state.keys():
                bias_weights = {
                    n: p.clone().detach() \
                    for n, p in params_state['bias'].items()
                }

            atten_weights = {}
            if 'atten' in params_state.keys():
                atten_weights = {
                    n: p.clone().detach() \
                    for n, p in params_state['atten'].items()
                }

            kb_weights = {}
            if 'aw_kb' in params_state.keys():
                kb_weights = {
                    n: p.clone().detach() \
                    for n, p in params_state['aw_kb'].items()
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
                **shared_weights,
                **adaptive_weights,
                **mask_weights,
                **bias_weights,
                **atten_weights,
                **kb_weights,
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

    def invoke_train(
            self,
            model: Model,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        train_acc = train_loss = 0.0
        batch_cnt = data_cnt = 0
        device = model.device

        model.train()
        self.set_optimizer_parameters(model)

        for data, person_id, classes_id in dataloader:
            data, target = data.to(device), person_id.to(device)
            self.optimizer.zero_grad()
            output = self._invoke_train(model, data, target, **kwargs)
            score, loss = output['score'], output['loss']
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
        l2_loss = lambda value: (value ** 2).sum()
        lambda_l1 = model.lambda_l1
        lambda_l2 = model.lambda_l2

        score, feature = model.forward(data)
        _loss = 0.0

        for loss_func in self.criterion:
            _loss += loss_func(score=score, feature=feature, target=target)

        d_layers = model.decomposed_module_leaves()
        sparseness = approx_loss = 0.0
        for name, module in d_layers:
            sparseness += torch.abs(module.aw).sum() + torch.abs(module.mask).sum()
            for _, pre_model in model.net_list.items():
                approx_loss += l2_loss(
                    (module.sw - model.net.get_submodule(name).sw) * module.mask + \
                    (module.aw - model.net.get_submodule(name).aw)
                ).sum()

        loss = _loss + lambda_l1 * sparseness + lambda_l2 * approx_loss

        return {
            'score': score,
            'feature': feature,
            'loss': loss
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
        features = []
        device = model.device

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
        features, labels = [], []
        device = model.device

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
            **kwargs
    ) -> None:
        super().__init__(client_name, model, operator, ckpt_root, **kwargs)
        self.current_task = None
        self.train_cnt = 0
        self.test_cnt = 0

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        self.model.update_model(params_state)

    def load_model(self, model_name: str):
        model_dict = self.model.model_state()
        model_dict = self.load_state(model_name, model_dict)
        self.model.update_model(model_dict)

    def save_model(self, model_name: str):
        model_dict = self.model.model_state()
        self.save_state(model_name, model_dict, True)

    def get_incremental_state(self, **kwargs) -> Dict:
        incremental_decomposed_layers = self.model.decomposed_module_leaves()
        incremental_adaptive_weights = {
            f'{name}.aw': layer.aw.clone().detach() \
            for name, layer in incremental_decomposed_layers
        }
        incremental_global_weights = {
            f'{name}.sw': (layer.mask * layer.sw + layer.aw +
                           torch.sum(layer.atten * layer.aw_kb, dim=-1, keepdim=False)
                           ).clone().detach() \
            for name, layer in incremental_decomposed_layers
        }
        return {
            'train_cnt': self.train_cnt,
            'incremental_aw': incremental_adaptive_weights,
            'incremental_gw': incremental_global_weights,
            'incremental_bn': self.model.model_state()['bn_params'],
        }

    def get_integrated_state(self, **kwargs) -> Dict:
        integrated_decomposed_layers = self.model.decomposed_module_leaves()
        integrated_adaptive_weights = {
            f'{name}.aw': layer.aw.clone().detach() \
            for name, layer in integrated_decomposed_layers
        }
        integrated_global_weights = {
            f'{name}.sw': (layer.mask * layer.sw + layer.aw +
                           torch.sum(layer.atten * layer.aw_kb, dim=-1, keepdim=False)
                           ).clone().detach() \
            for name, layer in integrated_decomposed_layers
        }
        return {
            'train_cnt': self.train_cnt,
            'integrated_aw': integrated_adaptive_weights,
            'integrated_gw': integrated_global_weights,
            'integrated_bn': self.model.model_state()['bn_params'],
            'pre_trained_params': self.model.model_state()['pre_trained_params'],
        }

    def update_by_incremental_state(self, state: Dict, **kwargs) -> Any:
        model_params = {
            'sw': state['incremental_sw'],
            'aw_kb': state['incremental_aw_kb'],
        }

        if self.current_task:
            self.load_model(self.current_task)
        self.update_model(model_params)
        for _, module in self.model.decomposed_module_leaves():
            module.aw.data = ((1.0 - module.mask.data) * module.sw.data).clone().detach()
        self.logger.info('Update model succeed by incremental state from server.')

    def update_by_integrated_state(self, state: Dict, **kwargs) -> Any:
        model_params = {
            'sw': state['integrated_sw'],
            'aw_kb': state['integrated_aw_kb'],
            'bn_params': state['integrated_bn'],
            'pre_trained_params': state['pre_trained_params'],
        }

        if self.current_task:
            self.load_model(self.current_task)
        self.update_model(model_params)
        for _, module in self.model.decomposed_module_leaves():
            module.aw.data = ((1.0 - module.mask.data) * module.sw.data).clone().detach()
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
        if self.current_task is not None and self.current_task != task_name:
            self.model.remember_params(task_name)
        self.current_task = task_name

        output = {}
        perf_loss, perf_acc, sustained_cnt = 1e8, 0, 0
        initial_lr = self.operator.optimizer.defaults['lr']

        with model_on_device(self.model, device):
            for epoch in range(1, epochs + 1):
                output = self.train_one_epoch(task_name, tr_loader, val_loader)
                accuracy, loss, data_count = output['accuracy'], output['loss'], output['data_count']

                # early stop
                sustained_cnt += 1
                if loss <= perf_loss and accuracy >= perf_acc:
                    perf_loss, perf_acc = loss, accuracy
                    sustained_cnt = 0
                if early_stop_threshold and sustained_cnt >= early_stop_threshold:
                    break

                self.train_cnt += data_count

                self.logger.info_train(
                    task_name, device,
                    data_count, perf_acc, perf_loss,
                    epoch, epochs
                )

        # Reset learning rate
        self.operator.optimizer.state = collections.defaultdict(dict)
        for param_group in self.operator.optimizer.param_groups:
            param_group['lr'] = initial_lr

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
            **kwargs
    ):
        super().__init__(server_name, model, operator, ckpt_root, **kwargs)
        self.client_aw = []

    def calculate(self) -> Any:
        merge_incremental_params = {}

        # calculate global weight and batch norm
        train_total_cnt = sum([client['train_cnt'] for _, client in self.clients.items()])
        for cname, cstate in self.clients.items():
            k, params = cstate['train_cnt'], {**cstate['incremental_gw'], **cstate['incremental_bn']}
            params = {
                n: (p.clone().detach() * k / train_total_cnt).type(dtype=p.dtype) \
                for n, p in params.items()
            }
            for n, p in params.items():
                if n not in merge_incremental_params.keys():
                    merge_incremental_params[n] = torch.zeros_like(p)
                merge_incremental_params[n] += p.clone().detach()

        # calculate knowledge base
        self.client_aw = []  # delete this line when need sample adaptive weight randomly as paper mentioned
        self.client_aw.extend([params['incremental_aw'] for _, params in self.clients.items()])
        if len(self.client_aw) >= self.model.kb_cnt:
            client_adaptive_weights = random.sample(self.client_aw, self.model.kb_cnt)
            for name, _ in client_adaptive_weights[0].items():
                merge_incremental_params[f'{name}_kb'] = torch.cat(
                    tensors=[aw[name].reshape(list(aw[name].shape) + [1]) \
                             for aw in client_adaptive_weights],
                    dim=-1
                )

        # load state for server model
        model_dict = self.model.net.state_dict()
        for i, (n, p) in enumerate(merge_incremental_params.items()):
            model_dict[n] = p.clone()
        self.model.net.load_state_dict(model_dict)

    def set_client_incremental_state(self, client_name: str, client_state: Dict) -> None:
        if client_name not in self.clients.keys():
            self.logger.warn(f'Collect incremental state failed from unregistered client {client_name}.')
        else:
            self.clients[client_name] = client_state
            self.logger.info(f'Collect incremental state successfully from client {client_name}.')

    def set_client_integrated_state(self, client_name: str, client_state: Dict) -> None:
        if client_name not in self.clients.keys():
            self.logger.warn(f'Collect integrated state failed from unregistered client {client_name}.')
        else:
            self.clients[client_name] = client_state
            self.logger.info(f'Collect integrated state successfully from client {client_name}.')

    def get_dispatch_incremental_state(self, client_name: str) -> Dict:
        model_state = self.model.model_state()
        return {
            'incremental_sw': model_state['sw'],
            'incremental_aw_kb': model_state['aw_kb'],
        }

    def get_dispatch_integrated_state(self, client_name: str) -> Dict:
        model_state = self.model.model_state()
        return {
            'integrated_sw': model_state['sw'],
            'integrated_aw_kb': model_state['aw_kb'],
            'integrated_bn': model_state['bn_params'],
            'pre_trained_params': model_state['pre_trained_params'],
        }
