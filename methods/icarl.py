import collections
from typing import Union, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.datasets_loader import ReIDImageDataset
from modules.client import ClientModule
from modules.model import ModelModule
from modules.operator import OperatorModule
from modules.server import ServerModule
from tools.evaluate import calculate_similarity_distance, evaluate
from tools.utils import model_on_device, get_one_hot


class Model(ModelModule):

    def __init__(
            self,
            net: Union[nn.Sequential, nn.Module],
            operator: OperatorModule = None,
            k: float = 8000,
            n_classes=1000,
            **kwargs
    ):
        super(Model, self).__init__(net)

        self.operator = operator
        self.k = k
        self.n_classes = n_classes
        self.arg = kwargs

        self.examplars = {}
        self.means = {}
        self.previous_logits = []

        self.features_extractor = self.net.base
        self.classifier = self.net.classifier

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)

    def add_n_classes(self, n):
        if n > 0:
            self.n_classes += n
            require_bias = self.classifier.bias is not None

            weight = self.classifier.weight.data
            bias = self.classifier.bias.data if require_bias else None

            self.classifier = nn.Linear(self.classifier.in_dim, self.n_classes, require_bias)

            self.classifier.weight.data[:self.n_classes - n] = weight
            if require_bias:
                self.classifier.bias.data[:self.n_classes - n] = bias

    def build_examplars(self, dataloader, device):
        imgs, ids, classes, features = [], [], [], []
        self.eval()
        for data, person_id, classes_id in dataloader:
            data = data.to(device)
            imgs.append(data.cpu())
            ids.append(person_id.cpu())
            classes.append(classes_id.cpu())
            features.append(self.features_extractor(data).cpu())

        imgs = torch.cat(imgs).detach().numpy()
        ids = torch.cat(ids).detach().numpy()
        classes = torch.cat(classes).detach().numpy()
        features = torch.cat(features).detach().numpy()

        for person_idx in np.unique(ids):
            _ids = np.argwhere(ids == person_idx).squeeze(axis=1)
            _imgs = imgs[_ids]
            _classes = classes[_ids]
            _features = features[_ids]
            _mean = _features / len(_features)

            examplars = []
            examplars_fea = []
            for i in range(self.k // self.n_classes):
                p = _mean - (_features + np.sum(examplars_fea, axis=0)) / (i + 1)
                p = np.linalg.norm(p, axis=1)
                min_idx = np.argmin(p)
                examplars.append((_imgs[min_idx], _classes[min_idx]))
                examplars_fea.append(_features[min_idx])

            self.examplars[person_idx] = examplars
            self.means[person_idx] = _mean

    def reduce_examplars(self):
        for class_idx in self.examplars.keys():
            self.examplars[class_idx] = self.examplars[class_idx][:self.k // self.n_classes]

    def model_state(self, *args, **kwargs) -> Dict:
        return {
            'net_params': {
                n: p.clone().detach() \
                for n, p in self.net.state_dict().items()
            },
            'examplars': {
                class_id: imgs.clone().detach() \
                for class_id, imgs in self.examplars.items()
            },
            'means': self.means
        }

    def update_model(self, params_state: Dict[str, torch.Tensor]):
        if 'net_params' in params_state.keys():
            net_dict = self.net.state_dict()
            for n, p in params_state['net_params'].items():
                net_dict[n] = p.clone().detach()
            self.net.load_state_dict(net_dict)

        if 'examplars' in params_state.keys():
            self.examplars = {class_id: imgs for class_id, imgs in params_state['examples'].items()}

        if 'means' in params_state.keys():
            self.means = params_state['means']


class Operator(OperatorModule):

    def set_optimizer_parameters(self, model: Model):
        optimizer_param_factory = {n: p for n, p in self.optimizer.defaults.items()}
        params = [p for p in model.net.parameters() if p.requires_grad]
        self.optimizer.param_groups = [{'params': params, **optimizer_param_factory}]

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
        model.add_n_classes(int(max(dataloader.dataset.person_ids)) - model.n_classes)
        self.set_optimizer_parameters(model)

        examplar_loader = DataLoader(ReIDImageDataset(source=model.examplars), batch_size=32)
        previous_logits = [model(data.to(device))[0].clone().detach().cpu() \
                           for data, person_id, classes_id in examplar_loader]

        # learn for incremental classes
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

        # learn and distill for memory images
        for idx, (data, person_id, classes_id) in enumerate(examplar_loader):
            data, target = data.to(device), person_id.to(device)
            self.optimizer.zero_grad()
            logit = model(data)[0]
            clf_loss = F.binary_cross_entropy_with_logits(
                input=logit,
                target=get_one_hot(target, model.n_classes).to(device)
            )
            distill_loss = F.binary_cross_entropy_with_logits(
                input=logit[:, :previous_logits[idx].shape[1]],
                target=torch.sigmoid(previous_logits[idx][:, :previous_logits[idx].shape[1]].to(device))
            )
            loss = clf_loss + distill_loss
            loss.backward()
            self.optimizer.step()

        model.reduce_examplars()
        model.build_examplars(dataloader, device)

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
