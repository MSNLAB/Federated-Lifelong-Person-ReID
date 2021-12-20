#######################################################
# Code from: https://github.com/HobbitLong/RepDistiller
#######################################################

import torch.nn.functional as F

from modules.criterion import CriterionModule


class DistillKL(CriterionModule):
    """Distilling the Knowledge in a Neural Network.
    Reference:
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>
    """

    def __init__(self, temperature: int = 1.0, **kwargs):
        super(DistillKL, self).__init__()
        for n, p in kwargs.items():
            self.__setattr__(n, p)
        self.temperature = temperature

    def forward(self, y_student, y_teacher):
        p_s = F.log_softmax(y_student / self.temperature, dim=1)
        p_t = F.softmax(y_teacher / self.temperature, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * \
               (self.temperature ** 2) / y_student.shape[0]
        return loss.sum()
