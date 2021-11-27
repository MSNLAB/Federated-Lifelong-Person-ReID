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

import copy
import random
from queue import Queue
from typing import Any, Dict, Union, List, Optional

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import Parameter
from torch.utils.data import DataLoader

from modules.client import ClientModule
from modules.operator import OperatorModule
from modules.server import ServerModule
from tools.evaluate import calculate_similarity_distance, evaluate
from tools.utils import torch_device, tensor_reverse_permute, model_on_device


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

        self.sw = Parameter(tensor_reverse_permute(shared_weight))
        self.sw.requires_grad = False

        self.bias = None
        if bias is not None:
            self.bias = Parameter(tensor_reverse_permute(bias))
            self.bias.requires_grad = True

        if mask is None:
            mask = torch.zeros(self.sw.shape[-1])
            mask = torch.sigmoid(mask)
        self.mask = Parameter(mask)
        self.mask.requires_grad = True

        if adaptive is None:
            adaptive = self.sw.clone().detach()
        self.aw = Parameter(adaptive)
        self.aw.requires_grad = True

        if knowledge_base is None:
            knowledge_base = torch.zeros(
                size=[shape for shape in self.sw.shape] + [kb_cnt]
            )
        self.aw_kb = Parameter(knowledge_base)
        self.aw_kb.requires_grad = False

        if atten is None:
            atten = torch.ones([kb_cnt])
            atten = torch.sigmoid(atten)
        self.atten = Parameter(atten)
        self.atten.requires_grad = True

        if lambda_l1 is None:
            lambda_l1 = 1e-3
        self.lambda_l1 = lambda_l1

        if lambda_mask is None:
            lambda_mask = 0.0
        self.lambda_mask = lambda_mask

    @staticmethod
    def l1_pruning(weights: torch.Tensor, hyper_parameters: torch.Tensor):
        hard_threshold = torch.greater(torch.abs(weights), hyper_parameters).float()
        return weights * hard_threshold

    def forward(self, data: torch.Tensor) -> Any:
        aw = self.aw if not self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask if not self.training else self.l1_pruning(self.mask, self.lambda_mask)
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
        aw = self.aw if not self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask if not self.training else self.l1_pruning(self.mask, self.lambda_mask)
        theta = mask * self.sw + aw + torch.sum(self.atten * self.aw_kb, dim=-1, keepdim=False)
        bias = self.bias
        return F.conv2d(
            input=data,
            weight=tensor_reverse_permute(theta),
            bias=tensor_reverse_permute(bias),
            stride=self.stride,
            padding=self.padding
        )


class Model(nn.Module):
    _decomposed_type_from = [nn.Linear, nn.Conv2d]
    _decomposed_type_to = [DecomposedLayer, DecomposedConv2D]

    def __init__(
            self,
            net: Union[nn.Sequential, nn.Module],
            lambda_l1: float = 1e-3,
            lambda_l2: float = 1e2,
            lambda_mask: float = 0.0,
            kb_cnt: int = 5,
            device: str = None,
            **kwargs
    ) -> None:
        super(Model, self).__init__()

        self.device = torch_device(device)
        self.net = net
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_mask = lambda_mask
        self.kb_cnt = kb_cnt
        self.args = kwargs
        self.model_list = {}
        self.layer_convert()

    def module_leaves(self) -> List:
        leaves = []
        q = Queue()
        for name, module in self.net.named_children():
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

    def decomposed_module_leaves(self) -> List:
        return [(name, module) for name, module in self.module_leaves() \
                if type(module) in self._decomposed_type_to]

    def remember_params(self, model_name: str):
        self.model_list[model_name] = copy.deepcopy(self.net)

    def layer_convert(self):
        leaves = self.module_leaves()
        for name, module in leaves:
            if type(module) not in self._decomposed_type_from:
                continue

            tuning_flag = False
            for param in module.parameters():
                if param.requires_grad:
                    tuning_flag = True
            if not tuning_flag:
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

            # replace module with decomposed layer
            pa_module = self.net
            name_path = name.split('.')
            for i, module_name in enumerate(name_path):
                if i + 1 == len(name_path):
                    pa_module.add_module(module_name, module)
                else:
                    pa_module = pa_module.__getattr__(module_name)

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
        bn_weight = {
            name: param.clone().detach() \
            for name, param in self.net.state_dict().items() \
            if 'bn' in name or 'bottleneck' in name or 'downsample' in name
        }

        return {
            'sw': shared_weights,
            'aw': adaptive_weights,
            'mask': mask_weights,
            'bias': bias_weights,
            'atten': atten_weights,
            'aw_kb': kb_weights,
            'bn': bn_weight
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

            bn_weights = {}
            if 'bn' in params_state.keys():
                bn_weights = {
                    n: p.clone().detach() \
                    for n, p in params_state['bn'].items()
                }

            model_params = {
                **shared_weights,
                **adaptive_weights,
                **mask_weights,
                **bias_weights,
                **atten_weights,
                **kb_weights,
                **bn_weights,
            }

            model_dict = self.net.state_dict()
            for i, (n, p) in enumerate(model_params.items()):
                model_dict[n] = p.clone().detach()
            self.net.load_state_dict(model_dict)


class Operator(OperatorModule):

    def set_optimizer_parameters(self, model: Model):
        optimizer_param_factory = {
            n: p for n, p in self.optimizer.param_groups[0].items() if n != 'params'
        }

        params = {}
        for name, param in model.net.base.named_parameters():
            if param.requires_grad:
                params[name] = param

        for name, param in model.net.classifier.named_parameters():
            if param.requires_grad:
                params[name] = param

        self.optimizer.param_groups = [{
            'params': params.values(),
            **optimizer_param_factory
        }]

    def invoke_train(
            self,
            model: Model,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        train_acc = train_loss = 0.0
        batch_cnt = data_cnt = 0

        model.train()
        self.set_optimizer_parameters(model)

        for data, person_id, classes_id in self.iter_dataloader(dataloader):
            data, target = data.to(self.device), classes_id.to(self.device)
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
            for _, pre_model in model.model_list.items():
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

        model.train()
        for data, person_id, classes_id in self.iter_dataloader(dataloader):
            data, target = data.to(self.device), classes_id.to(self.device)
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

        model.eval()
        for data, person_id, classes_id in self.iter_dataloader(dataloader):
            data = data.to(self.device)
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

        model.eval()
        for data, person_id, classes_id in self.iter_dataloader(dataloader):
            data, target = data.to(self.device), person_id.to(self.device)
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
            model: nn.Module,
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
            f'{name}.sw': (layer.mask * layer.sw).clone().detach() \
            for name, layer in incremental_decomposed_layers
        }

        return {
            'train_cnt': self.train_cnt,
            'increment_aw': incremental_adaptive_weights,
            'increment_gw': incremental_global_weights,
        }

    def get_integrated_state(self, **kwargs) -> Dict:
        integrated_decomposed_layers = self.model.decomposed_module_leaves()
        integrated_adaptive_weights = {
            f'{name}.aw': layer.aw.clone().detach() \
            for name, layer in integrated_decomposed_layers
        }
        integrated_global_weights = {
            f'{name}.sw': (layer.mask * layer.sw).clone().detach() \
            for name, layer in integrated_decomposed_layers
        }

        return {
            'train_cnt': self.train_cnt,
            'integrated_aw': integrated_adaptive_weights,
            'integrated_gw': integrated_global_weights,
        }

    def update_by_incremental_state(self, state: Dict, **kwargs) -> Any:
        model_params = {
            'sw': state['incremental_base_weights'],
            'aw_kb': state['incremental_knowledge_base']
        }

        if self.current_task:
            self.load_model(self.current_task)
        self.update_model(model_params)
        self.logger.info('Update model succeed by incremental state from server.')

    def update_by_integrated_state(self, state: Dict, **kwargs) -> Any:
        model_params = {
            'sw': state['integrated_base_weights'],
            'aw_kb': state['integrated_knowledge_base']
        }

        if self.current_task:
            self.load_model(self.current_task)
        self.update_model(model_params)
        self.logger.info('Update model succeed by integrated state from server.')

    def train(
            self,
            epochs: int,
            task_name: str,
            tr_loader: DataLoader,
            val_loader: DataLoader,
            early_stop_threshold: int = 3,
            **kwargs
    ) -> Any:
        if self.current_task is not None and self.current_task != task_name:
            self.model.remember_params(task_name)
        self.current_task = task_name

        output = {}
        perf_loss, perf_acc, sustained_cnt = 1e8, 0, 0
        initial_lr = self.operator.optimizer.defaults['lr']

        with model_on_device(self.model, self.device):
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
                    task_name, self.device,
                    data_count, perf_acc, perf_loss,
                    epoch, epochs
                )

        # Reset learning rate
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
            **kwargs
    ) -> Any:
        self.load_model(task_name)

        with model_on_device(self.model, self.device):
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
            **kwargs
    ) -> Any:
        self.load_model(task_name)

        with model_on_device(self.model, self.device):
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
            device=self.device
        )

        avg_representation = torch.cat([query_output['features'], gallery_output['features']], dim=0)
        avg_representation = torch.sum(avg_representation, dim=0) / len(avg_representation)

        self.logger.info_validation(task_name, query_size, gallery_size, cmc, mAP)
        return cmc, mAP, avg_representation


class Server(ServerModule):
    def __init__(
            self,
            server_name: str,
            model: nn.Module,
            operator: OperatorModule,
            ckpt_root: str,
            **kwargs
    ):
        super().__init__(server_name, model, operator, ckpt_root, **kwargs)
        self.client_aw = []

    def calculate(self) -> Any:
        merge_increment_params = {
            n: torch.zeros_like(p) \
            for n, p in self.model.net.state_dict().items()
        }

        # calculate global weight
        train_total_cnt = sum([client['train_cnt'] for _, client in self.clients.items()])
        for i, (cid, client) in enumerate(self.clients.items()):
            k, global_weight = client['train_cnt'], client['increment_gw']
            global_weight = {n: (p.clone().detach() * k / train_total_cnt).type(dtype=p.dtype) \
                             for n, p in global_weight.items()}
            for n, p in global_weight.items():
                merge_increment_params[n] += p.clone().detach()

        # calculate knowledge base
        self.client_aw = []  # delete this code if need shuffle adaptive weight as paper mentioned
        self.client_aw.extend([params['increment_aw'] for _, params in self.clients.items()])
        if len(self.client_aw) >= self.model.kb_cnt:
            client_adaptive_weights = random.sample(self.client_aw, self.model.kb_cnt)
            for name, _ in client_adaptive_weights[0].items():
                merge_increment_params[f'{name}_kb'] = torch.cat(
                    tensors=[aw[name].reshape(list(aw[name].shape) + [1]) \
                             for aw in client_adaptive_weights],
                    dim=-1
                )

        # load state for server model
        model_dict = self.model.net.state_dict()
        for i, (n, p) in enumerate(merge_increment_params.items()):
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

    def get_dispatch_incremental_state(self) -> Dict:
        incremental_decomposed_layers = self.model.decomposed_module_leaves()
        incremental_global_weights = {
            f'{name}.sw': layer.sw.clone().detach() \
            for name, layer in incremental_decomposed_layers
        }
        incremental_knowledge_base = {
            f'{name}.aw_kb': layer.aw_kb.clone().detach() \
            for name, layer in incremental_decomposed_layers
        }

        return {
            'incremental_base_weights': incremental_global_weights,
            'incremental_knowledge_base': incremental_knowledge_base,
        }

    def get_dispatch_integrated_state(self) -> Dict:
        integrated_decomposed_layers = self.model.decomposed_module_leaves()
        integrated_global_weights = {
            f'{name}.sw': layer.sw.clone().detach() \
            for name, layer in integrated_decomposed_layers
        }
        integrated_knowledge_base = {
            f'{name}.aw_kb': layer.aw_kb.clone().detach() \
            for name, layer in integrated_decomposed_layers
        }

        return {
            'integrated_base_weights': integrated_global_weights,
            'integrated_knowledge_base': integrated_knowledge_base,
        }
