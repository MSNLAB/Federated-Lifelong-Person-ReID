######################################################################################
# Code from: https://github.com/layumi/Person_reID_baseline_pytorch
# @article{zheng2019joint,
#   title={Joint discriminative and generative learning for person re-identification},
#   author={Zheng, Zhedong and Yang, Xiaodong and Yu, Zhiding and Zheng, and others},
#   journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#   year={2019}
# }
######################################################################################

import collections
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.utils.data import DataLoader

from modules.client import ClientModule
from modules.operator import OperatorModule
from modules.server import ServerModule
from tools.evaluate import calculate_similarity_distance, evaluate
from tools.utils import model_on_device


class Operator(OperatorModule):

    def invoke_train(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        train_acc = train_loss = 0.0
        batch_cnt = data_cnt = 0
        device = next(model.parameters()).device

        model.train()
        for data, person_id, classes_id in dataloader:
            data, target = data.to(device), classes_id.to(device)
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
            model: nn.Module,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        pred_acc = pred_loss = 0.0
        batch_cnt = data_cnt = 0
        device = next(model.parameters()).device

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
            'loss': loss
        }

    def invoke_inference(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        batch_cnt, data_cnt = 0, 0
        features = []
        device = next(model.parameters()).device

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
            model: nn.Module,
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
            model: nn.Module,
            dataloader: DataLoader,
            **kwargs
    ) -> Any:
        batch_cnt, data_cnt = 0, 0
        features, labels = [], []
        device = next(model.parameters()).device

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

    def update_by_incremental_state(self, state: Dict, **kwargs) -> Any:
        self.load_model(self.model_ckpt_name)
        self.update_model(state['model_params'])
        self.save_model(self.model_ckpt_name)
        self.logger.info('Update model succeed by integrated state from server.')

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
        model_ckpt_name = self.model_ckpt_name if self.model_ckpt_name else task_name
        self.load_model(model_ckpt_name)

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

        # Reset learning rate
        self.operator.optimizer.state = collections.defaultdict(dict)
        for param_group in self.operator.optimizer.param_groups:
            param_group['lr'] = initial_lr

        self.save_model(model_ckpt_name)
        return output

    def train_one_epoch(
            self,
            task_name: str,
            tr_loader: Union[List[DataLoader], DataLoader],
            val_loader: Union[List[DataLoader], DataLoader],
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
