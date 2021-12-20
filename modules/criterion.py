import torch.nn as nn


class CriterionModule(nn.Module):

    def __init__(self):
        super(CriterionModule, self).__init__()

    def forward(self, score, target, **kwargs):
        raise NotImplementedError
