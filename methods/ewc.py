#######################################################################################################
# Code from: https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW14/HW14.ipynb
# @article{kirkpatrick2017overcoming,
#   title={Overcoming catastrophic forgetting in neural networks},
#   author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, and others},
#   journal={Proceedings of the national academy of sciences},
#   year={2017},
#   url={https://arxiv.org/abs/1612.00796}
# }
#######################################################################################################

import collections
from typing import Any, Dict, Union, Optional, List

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.utils.data import DataLoader

from modules.client import ClientModule
from modules.model import ModelModule
from modules.operator import OperatorModule
from modules.server import ServerModule
from tools.evaluate import calculate_similarity_distance, evaluate
from tools.utils import model_on_device


class Model(ModelModule):

    def __init__(
            self,
            net: Union[nn.Sequential, nn.Module],
            operator: OperatorModule = None,
            lambda_penalty: float = 100.0,
            **kwargs
    ) -> None:
        super(Model, self).__init__(net)

        self.operator = operator
        self.lambda_penalty = lambda_penalty

        self.arg = kwargs
        self.params = {n: p for n, p in self.net.named_parameters() if p.requires_grad}
        self.params_old = {}
        self.precision_matrices = {}
        self.recall_dataloaders = {}

        self.calculate()

    def calculate(self) -> Dict[str, torch.Tensor]:
        self.precision_matrices = self._calculate_importance()
        for n, p in self.params.items():
            self.params_old[n] = p.clone().detach()
        return self.precision_matrices

    def _calculate_importance(self) -> Dict[str, torch.Tensor]:
        precision_matrices = {
            n: torch.zeros_like(p) \
            for n, p in self.params.items()
        }

        if len(self.recall_dataloaders) > 0:
            device = next(self.net.parameters()).device
            number_recall_data = sum(
                [len(loader) for task_name, loader in self.recall_dataloaders.items()]
            )
            for task_name, recall_dataloader in self.recall_dataloaders.items():
                for data, person_id, classes_id in recall_dataloader:
                    self.net.zero_grad()
                    data, target = data.to(device), classes_id.to(device)
                    loss = self.operator._invoke_train(self.net, data, target)['loss']
                    loss.backward()
                    for n, p in self.params.items():
                        precision_matrices[n].data += (p.grad.data ** 2) * len(data) \
                                                      / number_recall_data

        return precision_matrices

    def penalty(self) -> float:
        loss = 0.0
        for n, p in self.params.items():
            _loss = self.precision_matrices[n] * (p - self.params_old[n]) ** 2
            loss += _loss.sum()
        return self.lambda_penalty * loss

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)

    def remember_task(self, task_name: str, dataloader: DataLoader) -> None:
        self.recall_dataloaders[task_name] = dataloader
        self.calculate()

    def cpu(self) -> Any:
        self.net.cpu()
        for n, p in self.precision_matrices.items():
            self.precision_matrices[n] = p.cpu()
        for n, p in self.params_old.items():
            self.params_old[n] = p.cpu()
        return super().cpu()

    def cuda(self, device: Optional[Union[int, Any]] = None) -> Any:
        self.net.cuda()
        for n, p in self.precision_matrices.items():
            self.precision_matrices[n] = p.cuda()
        for n, p in self.params_old.items():
            self.params_old[n] = p.cuda()
        return super().cuda()

    def to(self, device: str = 'cpu'):
        self.net.to(device)
        for n, p in self.precision_matrices.items():
            self.precision_matrices[n] = p.to(device)
        for n, p in self.params_old.items():
            self.params_old[n] = p.to(device)
        return super().to(device)

    def model_state(self) -> Dict:
        return {
            'net_params': {
                n: p.clone().detach() \
                for n, p in self.net.state_dict().items()
            },
            'params_old': {
                n: p.clone().detach() \
                for n, p in self.params_old.items()
            },
            'precision_matrices': {
                n: p.clone().detach() \
                for n, p in self.precision_matrices.items()
            }
        }

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        if 'net_params' in params_state.keys():
            net_dict = self.net.state_dict()
            for n, p in params_state['net_params'].items():
                net_dict[n] = p.clone().detach()
            self.net.load_state_dict(net_dict)

        self.params = {
            n: p for n, p in self.net.named_parameters() \
            if p.requires_grad
        }

        if 'params_old' in params_state.keys():
            for n, p in self.params_old.items():
                self.params_old[n] = p.clone().detach()

        if 'precision_matrices' in params_state.keys():
            for n, p in self.precision_matrices.items():
                self.precision_matrices[n] = p.clone().detach()


class Operator(OperatorModule):

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
        for data, person_id, classes_id in dataloader:
            data, target = data.to(device), classes_id.to(device)
            self.optimizer.zero_grad()
            output = self._invoke_train(model, data, target, **kwargs)
            score, loss = output['score'], output['loss']
            losses = loss + model.penalty()
            losses.backward()
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
            data, target = data.to(device), classes_id.to(device)
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
            operator: OperatorModule,
            ckpt_root: str,
            model_ckpt_name: str = None,
            **kwargs
    ) -> None:
        super().__init__(client_name, model, operator, ckpt_root, model_ckpt_name, **kwargs)
        self.model.operator = operator
        if not self.model_ckpt_name:
            self.model_ckpt_name = 'ewc_model'

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        self.model.update_model(params_state)

    def load_model(self, model_name: str):
        model_dict = self.model.model_state()
        model_dict = self.load_state(model_name, model_dict)
        self.model.update_model(model_dict)

    def save_model(self, model_name: str):
        model_dict = self.model.model_state()
        self.save_state(model_name, model_dict, True)

    def update_by_incremental_state(self, state: Dict, **kwargs) -> Any:
        self.load_model(self.model_ckpt_name)
        self.update_model(state['model_params'])
        self.save_model(self.model_ckpt_name)
        self.logger.info('Update model succeed by incremental state from server.')

    def update_by_integrated_state(self, state: Dict, **kwargs) -> Any:
        self.load_model(self.model_ckpt_name)
        self.update_model(state['model_params'])
        self.save_model(self.model_ckpt_name)
        self.logger.info('Update model succeed by integrated state from server.')

    def train(
            self,
            epochs: int,
            task_name: str,
            tr_loader: Union[List[DataLoader], DataLoader],
            val_loader: Union[List[DataLoader], DataLoader],
            early_stop_threshold: int = 3,
            device: str = 'cpu',
            **kwargs
    ) -> Any:
        self.load_model(self.model_ckpt_name)

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

                self.logger.info_train(
                    task_name, device,
                    data_count, perf_acc, perf_loss,
                    epoch, epochs
                )

            self.model.remember_task(task_name, val_loader)

        # Reset learning rate
        self.operator.optimizer.state = collections.defaultdict(dict)
        for param_group in self.operator.optimizer.param_groups:
            param_group['lr'] = initial_lr

        self.save_model(self.model_ckpt_name)
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
        model_ckpt_name = self.model_ckpt_name if self.model_ckpt_name else task_name
        self.load_model(model_ckpt_name)

        with model_on_device(self.model, device):
            gallery_features = self.operator.invoke_inference(self.model, gallery_loader)['features']
            query_features = self.operator.invoke_inference(self.model, query_loader)['features']

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
        model_ckpt_name = self.model_ckpt_name if self.model_ckpt_name else task_name
        self.load_model(model_ckpt_name)

        with model_on_device(self.model, device):
            gallery_output = self.operator.invoke_valid(self.model, gallery_loader)
            query_output = self.operator.invoke_valid(self.model, query_loader)

        gallery_size = len(gallery_output['features'])
        query_size = len(query_output['features'])

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

    def get_dispatch_integrated_state(self, client_name: str) -> Dict:
        return {'model_params': {
            n: p.clone().detach() \
            for n, p in self.model.state_dict().items()
        }}
