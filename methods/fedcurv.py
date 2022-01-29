#############################################################################################
# @article{shoham2019overcoming,
#   title={Overcoming forgetting in federated learning on non-iid data},
#   author={Shoham, Neta and Avidor, Tomer and Keren, Aviv and Israel, Nadav, and others},
#   journal={arXiv preprint arXiv:1910.07796},
#   year={2019}
# }
#############################################################################################

import collections
from typing import Any, Dict, Union, Optional, List

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.modules.module import T
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
        self.other_precision_matrices = []
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
                    loss = self.operator._invoke_train(self, data, target)['loss']
                    loss.backward()
                    for n, p in self.params.items():
                        precision_matrices[n].data += (p.grad.data ** 2) * len(data) \
                                                      / number_recall_data

        return precision_matrices

    def penalty(self) -> float:
        loss = 0.0
        for n, p in self.params.items():
            _loss = (self.precision_matrices[n] * torch.abs(p - self.params_old[n]) ** 2).sum()
            for importance, params in self.other_precision_matrices:
                _loss += (importance[n] * torch.abs(p - params[n]) ** 2).sum()
            loss += _loss.sum()
        return self.lambda_penalty * loss

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)

    def remember_task(self, task_name: str, dataloader: DataLoader) -> None:
        self.recall_dataloaders[task_name] = dataloader
        self.calculate()

    def cpu(self: T) -> T:
        self.net.cpu()
        self.precision_matrices = {n: p.cpu() for n, p in self.precision_matrices.items()}
        self.params_old = {n: p.cpu() for n, p in self.params_old.items()}
        for idx, (importance, params) in enumerate(self.other_precision_matrices):
            self.other_precision_matrices[idx] = (
                {n: p.cpu() for n, p in importance.items()},
                {n: p.cpu() for n, p in params.items()}
            )
        return super().cpu()

    def cuda(self: T, device: Optional[Union[int, Any]] = None) -> T:
        self.net.cuda()
        self.precision_matrices = {n: p.cuda() for n, p in self.precision_matrices.items()}
        self.params_old = {n: p.cuda() for n, p in self.params_old.items()}
        for idx, (importance, params) in enumerate(self.other_precision_matrices):
            self.other_precision_matrices[idx] = (
                {n: p.cuda() for n, p in importance.items()},
                {n: p.cuda() for n, p in params.items()}
            )
        return super().cuda()

    def to(self, device: str = 'cpu'):
        self.net.to(device)
        self.precision_matrices = {n: p.to(device) for n, p in self.precision_matrices.items()}
        self.params_old = {n: p.to(device) for n, p in self.params_old.items()}
        for idx, (importance, params) in enumerate(self.other_precision_matrices):
            self.other_precision_matrices[idx] = (
                {n: p.to(device) for n, p in importance.items()},
                {n: p.to(device) for n, p in params.items()}
            )
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
            },
            'other_precision_matrices': [
                ({n: p.clone().detach() for n, p in importance.items()},
                 {n: p.clone().detach() for n, p in params.items()})
                for importance, params in self.other_precision_matrices
            ]
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

        if 'other_precision_matrices' in params_state.keys():
            self.other_precision_matrices = [
                ({n: p.clone().detach() for n, p in importance.items()},
                 {n: p.clone().detach() for n, p in params.items()})
                for importance, params in params_state['other_precision_matrices']
            ]


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
    ):
        super().__init__(client_name, model, operator, ckpt_root, model_ckpt_name, **kwargs)
        self.model.operator = operator
        if not self.model_ckpt_name:
            self.model_ckpt_name = 'fedcurv_model'
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
        increment_params = {
            n: p.clone().detach() \
            for n, p in self.model.net.named_parameters() \
            if p.requires_grad
        }

        increment_precision_matrices = {
            n: p.clone().detach() \
            for n, p in self.model.precision_matrices.items()
        }

        return {
            'train_cnt': self.train_cnt,
            'incremental_model_params': increment_params,
            'incremental_precision_matrices': increment_precision_matrices,
        }

    def get_integrated_state(self, **kwargs) -> Dict:
        integrated_params = {
            n: p.clone().detach() \
            for n, p in self.model.net.state_dict().items()
        }

        integrated_precision_matrices = {
            n: p.clone().detach() \
            for n, p in self.model.precision_matrices.items()
        }

        return {
            'train_cnt': self.train_cnt,
            'integrated_model_params': integrated_params,
            'integrated_precision_matrices': integrated_precision_matrices,
        }

    def update_by_incremental_state(self, state: Dict, **kwargs) -> Any:
        other_increment_params = state['other_clients_incremental_params']
        other_increment_matrices = state['other_clients_precision_matrices']
        other_precision_matrices = []
        for idx in range(len(other_increment_params)):
            other_precision_matrices.append(
                (other_increment_matrices[idx], other_increment_params[idx])
            )

        model_state = {
            'net_params': state['incremental_model_params'],
            'other_precision_matrices': other_precision_matrices
        }

        self.train_cnt = self.test_cnt = 0
        self.load_model(self.model_ckpt_name)
        self.update_model(model_state)
        self.save_model(self.model_ckpt_name)
        self.logger.info('Update model succeed by incremental state from server.')

    def update_by_integrated_state(self, state: Dict, **kwargs) -> Any:
        other_increment_params = state['other_clients_integrated_params']
        other_increment_matrices = state['other_clients_precision_matrices']
        other_precision_matrices = []
        for idx in range(len(other_increment_params)):
            other_precision_matrices.append(
                (other_increment_params[idx], other_increment_matrices[idx])
            )

        model_state = {
            'net_params': state['integrated_model_params'],
            'other_precision_matrices': other_precision_matrices
        }

        self.train_cnt = self.test_cnt = 0
        self.load_model(self.model_ckpt_name)
        self.update_model(model_state)
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

                self.train_cnt += data_count

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
        self.load_model(self.model_ckpt_name)

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
        self.load_model(self.model_ckpt_name)

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

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        self.model.update_model(params_state)

    def calculate(self) -> Any:
        merge_increment_params = {}
        train_total_cnt = sum([client['train_cnt'] for _, client in self.clients.items()])

        for _, client in self.clients.items():
            k, params = client['train_cnt'], client['incremental_model_params']
            for i, (n, p) in enumerate(params.items()):
                if n not in merge_increment_params.keys():
                    merge_increment_params[n] = torch.zeros_like(p)
                merge_increment_params[n] += (p.clone().detach() * k / train_total_cnt).type(dtype=p.dtype)

        model_state = {'net_params': merge_increment_params}

        self.update_model(model_state)

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
        incremental_model_params = {
            n: p.clone().detach() \
            for n, p in self.model.net.named_parameters()
            if p.requires_grad
        }

        other_clients_incremental_params = [
            {
                n: p.clone().detach() \
                for n, p in state['incremental_model_params'].items()
            } for client, state in self.clients.items() if state is not None
        ]

        other_clients_precision_matrices = [
            {
                n: p.clone().detach() \
                for n, p in state['incremental_precision_matrices'].items()
            } for client, state in self.clients.items() if state is not None
        ]

        return {
            'incremental_model_params': incremental_model_params,
            'other_clients_incremental_params': other_clients_incremental_params,
            'other_clients_precision_matrices': other_clients_precision_matrices,
        }

    def get_dispatch_integrated_state(self, client_name: str) -> Dict:
        integrated_model_params = {
            n: p.clone().detach() \
            for n, p in self.model.net.state_dict().items()
        }

        other_clients_integrated_params = [
            {
                n: p.clone().detach() \
                for n, p in state['integrated_model_params'].items()
            } for client, state in self.clients.items() if state is not None
        ]

        other_clients_precision_matrices = [
            {
                n: p.clone().detach() \
                for n, p in state['integrated_precision_matrices'].items()
            } for client, state in self.clients.items() if state is not None
        ]

        return {
            'integrated_model_params': integrated_model_params,
            'other_clients_integrated_params': other_clients_integrated_params,
            'other_clients_precision_matrices': other_clients_precision_matrices,
        }
