####################################################################
# Code from : https://github.com/michuanhaohao/reid-strong-baseline
####################################################################

import torch.nn as nn


def weights_init_kaiming(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
        nn.init.constant_(module.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if module.affine:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


def weights_init_classifier(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(module.weight, std=0.001)
        if module.bias:
            nn.init.constant_(module.bias, 0.0)
