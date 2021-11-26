import copy
from math import e
from queue import Queue
from typing import Any, Dict, Union, List, Optional

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import Parameter
from torch.utils.data import DataLoader

from criterions.kd_loss import DistillKL
from modules.client import ClientModule
from modules.operator import OperatorModule
from modules.server import ServerModule
from tools.evaluate import calculate_similarity_distance, evaluate
from tools.utils import torch_device, tensor_reverse_permute, model_on_device, normalize


class DecomposedLayer(nn.Module):

    def __init__(
            self,
            global_weight: torch.Tensor,
            global_weight_atten: torch.Tensor = None,
            adaptive_weight: torch.Tensor = None,
            adaptive_bias: torch.Tensor = None,
            lambda_l1: float = None,
            lambda_atten: float = None,
            **kwargs
    ):
        super(DecomposedLayer, self).__init__()

        self.global_weight = Parameter(tensor_reverse_permute(global_weight))
        self.global_weight.requires_grad = False

        if global_weight_atten is None:
            global_weight_atten = torch.ones(self.global_weight.shape[-1])
            global_weight_atten = torch.sigmoid(global_weight_atten * e)
        self.global_weight_atten = Parameter(global_weight_atten)
        self.global_weight_atten.requires_grad = True

        if adaptive_weight is None:
            adaptive_weight = self.global_weight.data.clone().detach()
            adaptive_weight = (1 - global_weight_atten.data) * adaptive_weight
        self.adaptive_weight = Parameter(adaptive_weight)
        self.adaptive_weight.requires_grad = True

        self.adaptive_bias = None
        if adaptive_bias is not None:
            self.adaptive_bias = Parameter(tensor_reverse_permute(adaptive_bias))
            self.adaptive_bias.requires_grad = True

        if lambda_l1 is None:
            lambda_l1 = 1e-3
        self.lambda_l1 = lambda_l1

        if lambda_atten is None:
            lambda_atten = 0.0
        self.lambda_atten = lambda_atten

    @staticmethod
    def l1_pruning(weights: torch.Tensor, hyper_parameters: torch.Tensor):
        hard_threshold = torch.greater(torch.abs(weights), hyper_parameters).float()
        return weights * hard_threshold

    def forward(self, data: torch.Tensor) -> Any:
        adaptive_weight = self.adaptive_weight
        adaptive_bias = self.adaptive_bias
        global_weight = self.global_weight
        global_weight_atten = self.global_weight_atten

        # if self.training:
        #     adaptive_weight = self.l1_pruning(self.adaptive_weight, self.lambda_l1)
        #     adaptive_bias = self.l1_pruning(self.adaptive_bias, self.lambda_l1) if adaptive_bias else None
        #     global_weight_atten = self.l1_pruning(self.global_weight_atten, self.lambda_atten)

        theta = global_weight_atten * global_weight + adaptive_weight
        bias = adaptive_bias

        return F.linear(
            input=data,
            weight=tensor_reverse_permute(theta),
            bias=tensor_reverse_permute(bias)
        )


class DecomposedConv2D(DecomposedLayer):

    def __init__(
            self,
            global_weight: torch.Tensor,
            adaptive_weight: torch.Tensor = None,
            adaptive_bias: torch.Tensor = None,
            global_weight_atten: torch.Tensor = None,
            lambda_l1: float = None,
            lambda_atten: float = None,
            stride: int = 1,
            padding: int = 0,
            **kwargs
    ) -> None:
        super().__init__(
            global_weight,
            adaptive_weight,
            adaptive_bias,
            global_weight_atten,
            lambda_l1,
            lambda_atten,
            **kwargs
        )
        self.stride = stride
        self.padding = padding

    def forward(self, data: torch.Tensor) -> Any:
        adaptive_weight = self.adaptive_weight
        adaptive_bias = self.adaptive_bias
        global_weight = self.global_weight
        global_weight_atten = self.global_weight_atten

        # if self.training:
        #     adaptive_weight = self.l1_pruning(self.adaptive_weight, self.lambda_l1)
        #     adaptive_bias = self.l1_pruning(self.adaptive_bias, self.lambda_l1) if adaptive_bias else None
        #     global_weight_atten = self.l1_pruning(self.global_weight_atten, self.lambda_atten)

        theta = global_weight_atten * global_weight + adaptive_weight
        bias = adaptive_bias

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
            lambda_kd_1: float = 0.0,
            lambda_kd_2: float = 0.0,
            lambda_atten: float = 1e-3,
            device: str = None,
            **kwargs
    ) -> None:
        super(Model, self).__init__()

        self.device = torch_device(device)
        self.net = net
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_kd_1 = lambda_kd_1
        self.lambda_kd_2 = lambda_kd_2
        self.lambda_atten = lambda_atten
        self.args = kwargs

        self.layer_convert()

    def layer_convert(self):
        leaves = self.module_leaves(self.net.base)
        for name, module in leaves:
            # pass through modules not like decomposed type
            if type(module) not in self._decomposed_type_from:
                continue

            # pass through modules no need to train
            tuning_flag = False
            for param in module.parameters():
                if param.requires_grad:
                    tuning_flag = True
            if not tuning_flag:
                continue

            if isinstance(module, nn.Linear):
                module = DecomposedLayer(
                    global_weight=module.weight,
                    adaptive_bias=module.bias,
                    lambda_l1=self.lambda_l1,
                    lambda_atten=self.lambda_atten
                )

            if isinstance(module, nn.Conv2d):
                module = DecomposedConv2D(
                    global_weight=module.weight,
                    adaptive_bias=module.bias,
                    lambda_l1=self.lambda_l1,
                    lambda_atten=self.lambda_atten,
                    stride=module.stride,
                    padding=module.padding,
                )

            # replace module with decomposed layer
            pa_module = self.net.base
            name_path = name.split('.')
            for deep, module_name in enumerate(name_path, 1):
                if deep == len(name_path):
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

    def decomposed_module_leaves(self) -> List:
        return [(name, module) for name, module in self.module_leaves(self.net) \
                if type(module) in self._decomposed_type_to]

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

        global_weights = {
            f'{name}.global_weight': layer.global_weight.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.global_weight is not None
        }
        atten = {
            f'{name}.global_weight_atten': layer.global_weight_atten.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.global_weight_atten is not None
        }
        adaptive_weights = {
            f'{name}.adaptive_weight': layer.adaptive_weight.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.adaptive_weight is not None
        }
        adaptive_bias = {
            f'{name}.adaptive_bias': layer.adaptive_bias.clone().detach() \
            for name, layer in decomposed_layers \
            if layer.adaptive_bias is not None
        }
        bn_weight = {
            name: params.clone().detach() \
            for name, params in self.net.state_dict().items() \
            if 'bn' in name or 'bottleneck' in name or 'downsample' in name
        }
        classifier_weight = {
            name: params.clone().detach() \
            for name, params in self.net.state_dict().items() \
            if 'classifier' in name
        }

        return {
            'global_weight': global_weights,
            'global_weight_atten': atten,
            'adaptive_weights': adaptive_weights,
            'adaptive_bias': adaptive_bias,
            'bn_weight': bn_weight,
            'classifier_weight': classifier_weight
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

        bn_weight = {}
        if 'bn_weight' in params_state.keys():
            bn_weight = {
                n: p.clone().detach() \
                for n, p in params_state['bn_weight'].items()
            }

        classifier_weight = {}
        if 'classifier_weight' in params_state.keys():
            classifier_weight = {
                n: p.clone().detach() \
                for n, p in params_state['classifier_weight'].items()
            }

        model_params = {
            **global_weight,
            **global_weight_atten,
            **adaptive_weights,
            **adaptive_bias,
            **bn_weight,
            **classifier_weight,
        }

        model_dict = self.net.state_dict()
        for i, (n, p) in enumerate(model_params.items()):
            model_dict[n] = p.clone().detach()
        self.net.load_state_dict(model_dict)


class Operator(OperatorModule):

    def set_optimizer_train_classifier(self, model: Model):
        optimizer_param_factory = {
            n: p for n, p in self.optimizer.param_groups[0].items() if n != 'params'
        }

        params = []

        for name, param in model.net.classifier.named_parameters():
            if param.requires_grad:
                params.append(param)

        self.optimizer.param_groups = [{
            'params': params,
            **optimizer_param_factory
        }]

    def set_optimizer_train_backbone(self, model: Model):
        optimizer_param_factory = {
            n: p for n, p in self.optimizer.param_groups[0].items() if n != 'params'
        }

        params = []

        for name, param in model.net.base.named_parameters():
            if param.requires_grad:
                params.append(param)

        self.optimizer.param_groups = [{
            'params': params,
            **optimizer_param_factory
        }]

    def set_optimizer_train_all(self, model: Model):
        optimizer_param_factory = {
            n: p for n, p in self.optimizer.param_groups[0].items() if n != 'params'
        }

        params = []
        for name, param in model.net.named_parameters():
            if param.requires_grad:
                params.append(param)

        self.optimizer.param_groups = [{
            'params': params,
            **optimizer_param_factory
        }]

    def invoke_train(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        train_acc = train_loss = 0.0
        batch_cnt = data_cnt = 0

        model.train()
        self.set_optimizer_train_all(model)

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

    def _invoke_train_kd(
            self,
            model: Model,
            pre_model: Model,
            kd_models: List[Model],
            dataloader: DataLoader,
            **kwargs
    ):
        train_acc = train_loss = 0.0
        batch_cnt = data_cnt = 0

        kd_loss = DistillKL(2.0)

        model.train()
        self.set_optimizer_train_all(model)

        for data, person_id, classes_id in self.iter_dataloader(dataloader):
            data, target = data.to(self.device), classes_id.to(self.device)
            self.optimizer.zero_grad()
            stu_output = self._invoke_train(model, data, target, **kwargs)
            score, feature, loss = stu_output['score'], stu_output['feature'], stu_output['loss']
            y_student = F.normalize(feature)

            # knowledge from relevant models
            y_teacher = torch.zeros_like(y_student)
            y_teacher_count = 0
            for kd_model in kd_models:
                kd_model.train()
                with model_on_device(kd_model, self.device):
                    with torch.no_grad():
                        tea_output = self._invoke_predict(kd_model, data, target, **kwargs)
                    t_score, t_feature = tea_output['score'], tea_output['feature']
                    y_teacher += F.normalize(t_feature.clone().detach())
                    y_teacher_count += 1

            if y_teacher_count:
                y_teacher /= y_teacher_count
                # y_teacher += (torch.randn(y_teacher.shape) >> 10).to(y_teacher.device)  # add noise
                loss += kd_loss(y_student, y_teacher) * model.lambda_kd_2

            # knowledge from pre-model
            if pre_model:
                pre_model.train()
                with model_on_device(pre_model, self.device):
                    with torch.no_grad():
                        tea_output = self._invoke_predict(pre_model, data, target, **kwargs)
                    t_score, t_feature = tea_output['score'], tea_output['feature']
                    loss += kd_loss(y_student, F.normalize(t_feature.clone().detach())) \
                            * model.lambda_kd_1

            # l1 to sparse the adaptive weight
            d_layers = model.decomposed_module_leaves()
            sparseness = 0.0
            for name, module in d_layers:
                sparseness += torch.abs(module.adaptive_weight).sum()
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
            'accuracy': train_acc,
            'loss': train_loss,
            'batch_count': batch_cnt,
            'data_count': data_cnt,
        }

    def _invoke_train(
            self,
            model: nn.Module,
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
            model_ckpt_name: str = None,
            **kwargs
    ) -> None:
        super().__init__(client_name, model, operator, ckpt_root, model_ckpt_name, **kwargs)
        self.current_task = None
        self.current_convergence = 0.0
        self.train_cnt = 0
        self.test_cnt = 0
        self.pre_model = None
        self.kd_models = []

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

        model_params = {
            n: p.clone().detach() \
            for n, p in self.model.net.state_dict().items()
        }

        incremental_shared_weights = {
            f'{name}.global_weight': layer.global_weight_atten * layer.global_weight \
                                     + layer.adaptive_weight \
            for name, layer in incremental_decomposed_layers \
            if layer.global_weight_atten is not None
        }

        return {
            'train_cnt': self.train_cnt,
            'model_convergence': self.current_convergence,
            'model_params': model_params,
            'incremental_shared_weight': incremental_shared_weights,
        }

    def get_integrated_state(self, **kwargs) -> Dict:
        integrated_decomposed_layers = self.model.decomposed_module_leaves()

        model_params = {
            n: p.clone().detach() \
            for n, p in self.model.net.state_dict().items()
        }

        integrated_shared_weights = {
            f'{name}.global_weight': layer.global_weight_atten * layer.global_weight \
                                     + layer.adaptive_weight
            for name, layer in integrated_decomposed_layers
        }

        return {
            'train_cnt': self.train_cnt,
            'model_convergence': self.current_convergence,
            'model_params': model_params,
            'integrated_shared_weight': integrated_shared_weights,
        }

    def update_by_incremental_state(self, state: Dict, **kwargs) -> Any:
        model_params = {
            'global_weight': state['incremental_global_weight'],
            'bn_weight': state['incremental_bn_weight'],
            'classifier_weight': state['incremental_classifier_weight'],
        }

        self.kd_models = []
        for relevant_model_params in state['relevant_model_params']:
            copy_model = copy.deepcopy(self.model)

            model_dict = self.model.net.state_dict()
            for i, (n, p) in enumerate(relevant_model_params.items()):
                model_dict[n] = p.clone().detach()
            self.model.net.load_state_dict(model_dict)
            self.kd_models.append(copy_model)

        for idx, relevant_model_convergence in enumerate(state['relevant_model_convergence']):
            self.kd_models[idx].lambda_kd_2 *= relevant_model_convergence

        if self.current_task:
            if self.model_ckpt_name:
                self.load_model(self.model_ckpt_name)
            else:
                self.load_model(self.current_task)
        self.update_model(model_params)
        self.logger.info('Update model succeed by incremental state from server.')

    def update_by_integrated_state(self, state: Dict, **kwargs) -> Any:
        model_params = {
            'global_weight': state['integrated_global_weight'],
            'bn_weight': state['integrated_bn_weight'],
            'classifier_weight': state['integrated_classifier_weight'],
        }

        self.kd_models = []
        if self.current_task:
            if self.model_ckpt_name:
                self.load_model(self.model_ckpt_name)
            else:
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
            self.pre_model = copy.deepcopy(self.model)
            self.current_convergence = 0.0
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
        self.current_convergence = accuracy

        # Reset learning rate
        for param_group in self.operator.optimizer.param_groups:
            param_group['lr'] = initial_lr

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
        if self.pre_model is not None and len(self.kd_models) == 0:
            return self.operator.invoke_train(self.model, tr_loader)
        else:
            return self.operator._invoke_train_kd(
                self.model, self.pre_model,
                self.kd_models, tr_loader
            )

    def inference(
            self,
            task_name: str,
            query_loader: Union[List[DataLoader], DataLoader],
            gallery_loader: Union[List[DataLoader], DataLoader],
            **kwargs
    ) -> Any:
        if self.model_ckpt_name:
            self.load_model(self.model_ckpt_name)
        else:
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
        if self.model_ckpt_name:
            self.load_model(self.model_ckpt_name)
        else:
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

        self.logger.info_validation(task_name, query_size, gallery_size, cmc, mAP)
        return cmc, mAP


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
        self.client_shared_weight = []

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
        merge_increment_params = {}

        train_total_cnt = sum([client['train_cnt'] for _, client in self.clients.items()])

        for i, (cid, client) in enumerate(self.clients.items()):
            k, global_weight = client['train_cnt'], client['model_params']
            global_weight = {
                n: (p.clone().detach() * k / train_total_cnt).type(dtype=p.dtype) \
                for n, p in global_weight.items() \
                if 'bn' in n or 'bottleneck' in n or 'downsample' in n
            }
            for n, p in global_weight.items():
                if n not in merge_increment_params.keys():
                    merge_increment_params[n] = torch.zeros_like(p)
                merge_increment_params[n] += p.clone().detach()

        for i, (cid, client) in enumerate(self.clients.items()):
            k, global_weight = client['train_cnt'], client['incremental_shared_weight']
            global_weight = {n: (p.clone().detach() * k / train_total_cnt).type(dtype=p.dtype) \
                             for n, p in global_weight.items()}
            for n, p in global_weight.items():
                if n not in merge_increment_params.keys():
                    merge_increment_params[n] = torch.zeros_like(p)
                merge_increment_params[n] += p.clone().detach()

        model_dict = self.model.net.state_dict()
        for i, (n, p) in enumerate(merge_increment_params.items()):
            model_dict[n] = p.clone().detach()
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

        incremental_global_weight = {
            f'{name}.global_weight': layer.global_weight.clone().detach() \
            for name, layer in self.model.decomposed_module_leaves()
        }

        incremental_bn_weight = {
            n: p.clone().detach() \
            for n, p in self.model.net.state_dict().items() \
            if 'bn' in n or 'bottleneck' in n or 'downsample' in n
        }

        incremental_classifier_weight = {
            n: p.clone().detach() \
            for n, p in self.model.net.state_dict().items() \
            if 'classifier' in n
        }

        relevant_model_convergence = [
            client['model_convergence'] \
            for _, client in self.clients.items() \
            if client and client['model_convergence']
        ]

        relevant_model_convergence = \
            normalize(relevant_model_convergence).tolist()

        relevant_models = [
            client['model_params'] \
            for _, client in self.clients.items() \
            if client and client['model_convergence']
        ]

        return {
            'incremental_global_weight': incremental_global_weight,
            'incremental_bn_weight': incremental_bn_weight,
            'incremental_classifier_weight': incremental_classifier_weight,
            'relevant_model_convergence': relevant_model_convergence,
            'relevant_model_params': relevant_models
        }

    def get_dispatch_integrated_state(self) -> Dict:

        integrated_global_weight = {
            f'{name}.global_weight': layer.global_weight.clone().detach() \
            for name, layer in self.model.decomposed_module_leaves()
        }

        integrated_bn_weight = {
            n: p.clone().detach() \
            for n, p in self.model.net.state_dict().items() \
            if 'bn' in n or 'bottleneck' in n or 'downsample' in n
        }

        integrated_classifier_weight = {
            n: p.clone().detach() \
            for n, p in self.model.net.state_dict().items() \
            if 'classifier' in n
        }

        relevant_model_convergence = [
            client['model_convergence'] \
            for _, client in self.clients.items() \
            if client and client['model_convergence']
        ]

        relevant_model_convergence = \
            normalize(relevant_model_convergence).tolist()

        relevant_models = [
            client['model_params'] \
            for _, client in self.clients.items() \
            if client and client['model_convergence']
        ]

        return {
            'integrated_global_weight': integrated_global_weight,
            'integrated_bn_weight': integrated_bn_weight,
            'integrated_classifier_weight': integrated_classifier_weight,
            'relevant_model_convergence': relevant_model_convergence,
            'relevant_model_params': relevant_models
        }
