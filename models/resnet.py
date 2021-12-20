####################################################################
# Code from : https://github.com/michuanhaohao/reid-strong-baseline
####################################################################

from typing import Type, Union, List, Optional, Callable

import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url

from tools.winit import weights_init_kaiming, weights_init_classifier

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}


def _conv3x3(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def _conv1x1(
        in_planes: int,
        out_planes: int,
        stride: int = 1
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[_BasicBlock, _Bottleneck]],
            layers: List[int],
            last_stride: int = 2,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, _Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, _BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self,
            block: Type[Union[_BasicBlock, _Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNet_ReID(nn.Module):

    def __init__(
            self,
            model_name: str,
            num_classes: int = 1000,
            last_stride: int = 2,
            neck: str = 'no',
            **kwargs
    ) -> None:
        super(ResNet_ReID, self).__init__()

        for n, p in kwargs.items():
            self.__setattr__(n, p)

        self.model_name = model_name
        self.num_classes = num_classes
        self.neck = neck

        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = _ResNet(last_stride=last_stride,
                                block=_BasicBlock,
                                layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = _ResNet(last_stride=last_stride,
                                block=_BasicBlock,
                                layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.in_planes = 2048
            self.base = _ResNet(last_stride=last_stride,
                                block=_Bottleneck,
                                layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.in_planes = 2048
            self.base = _ResNet(last_stride=last_stride,
                                block=_Bottleneck,
                                layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.in_planes = 2048
            self.base = _ResNet(last_stride=last_stride,
                                block=_Bottleneck,
                                layers=[3, 8, 36, 3])
        else:
            raise ValueError(f'No model named {model_name} for generating.')

        self.gap = nn.AdaptiveAvgPool2d(1)

        if neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
        elif neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        else:
            raise ValueError(f'Mismatched neck type for {neck}.')

        base_state_dict = load_state_dict_from_url(model_urls[model_name], progress=False)
        del base_state_dict['fc.weight'], base_state_dict['fc.bias'],
        self.base.load_state_dict(base_state_dict)

    def forward(self, x):
        global_feat = self.gap(self.base(x))
        global_feat = global_feat.view(global_feat.shape[0], -1)

        if self.neck == 'bnneck' and len(x) > 1:
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        else:
            feat = global_feat

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            return global_feat


def resnet18(**kwargs):
    return ResNet_ReID(model_name='resnet18', **kwargs)


def resnet34(**kwargs):
    return ResNet_ReID(model_name='resnet34', **kwargs)


def resnet50(**kwargs):
    return ResNet_ReID(model_name='resnet50', **kwargs)


def resnet101(**kwargs):
    return ResNet_ReID(model_name='resnet101', **kwargs)


def resnet152(**kwargs):
    return ResNet_ReID(model_name='resnet152', **kwargs)
