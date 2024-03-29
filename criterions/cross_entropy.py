###################################################
# Code from: https://github.com/JDAI-CV/fast-reid/
###################################################

import torch
import torch.nn as nn

from modules.criterion import CriterionModule


class CrossEntropyLabelSmooth(CriterionModule):
    """Cross entropy loss with label smoothing regularized.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, **kwargs):
        super(CrossEntropyLabelSmooth, self).__init__()
        for n, p in kwargs.items():
            self.__setattr__(n, p)
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, score, target, **kwargs):
        """
        Args:
            score: prediction matrix (before softmax) with shape (batch_size, num_classes)
            target: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(score)
        target = torch.zeros(log_probs.size()) \
            .scatter_(1, target.unsqueeze(1).data.cpu(), 1) \
            .to(log_probs.device)
        target = (1 - self.epsilon) * target + self.epsilon / self.num_classes
        loss = (- target * log_probs).mean(0).sum()
        return loss
