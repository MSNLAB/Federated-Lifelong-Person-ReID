###################################################
# Code from: https://github.com/JDAI-CV/fast-reid/
###################################################

import torch
import torch.nn.functional as F

from modules.criterion import CriterionModule
from tools.distance import compute_cosine_distance, compute_euclidean_distance


class TripletLoss(CriterionModule):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Reference:
    Alexander Hermans et al. In Defense of the Triplet Loss for Person Re-Identification.
    """

    def __init__(self, margin=None, norm_feat=False, hard_mining=False, **kwargs):
        super(TripletLoss, self).__init__()
        for n, p in kwargs.items():
            self.__setattr__(n, p)
        self.margin = margin
        self.norm_feat = norm_feat
        self.hard_mining = hard_mining

    @staticmethod
    def softmax_weights(dist, mask):
        max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
        diff = dist - max_v
        Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
        W = torch.exp(diff) * mask / Z
        return W

    @staticmethod
    def hard_example_mining(dist_mat, is_pos, is_neg):
        """For each anchor, find the hardest positive and negative sample.
        Args:
          dist_mat: pair wise distance between samples, shape [N, M]
          is_pos: positive index with shape [N, M]
          is_neg: negative index with shape [N, M]
        Returns:
          dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
          dist_an: pytorch Variable, distance(anchor, negative); shape [N]
          p_inds: pytorch LongTensor, with shape [N];
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
          n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        NOTE: Only consider the case in which all labels have same num of samples,
          thus we can cope with all anchors in parallel.
        """

        assert len(dist_mat.size()) == 2

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N]
        dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N]
        dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

        return dist_ap, dist_an

    @staticmethod
    def weighted_example_mining(dist_mat, is_pos, is_neg):
        """For each anchor, find the weighted positive and negative sample.
        Args:
          dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
          is_pos:
          is_neg:
        Returns:
          dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
          dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        """
        assert len(dist_mat.size()) == 2

        is_pos = is_pos
        is_neg = is_neg
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = TripletLoss.softmax_weights(dist_ap, is_pos)
        weights_an = TripletLoss.softmax_weights(-dist_an, is_neg)

        dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
        dist_an = torch.sum(dist_an * weights_an, dim=1)

        return dist_ap, dist_an

    def __call__(self, feature, target, normalize_feature=False, **kwargs):
        r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
            Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
            Loss for Person Re-Identification'."""

        if self.norm_feat:
            dist_mat = compute_cosine_distance(feature, feature)
        else:
            dist_mat = compute_euclidean_distance(feature, feature)

        # For distributed training, gather all features from different process.
        # if comm.get_world_size() > 1:
        #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
        #     all_targets = concat_all_gather(targets)
        # else:
        #     all_embedding = embedding
        #     all_targets = targets

        N = dist_mat.size(0)
        is_pos = target.view(N, 1).expand(N, N).eq(target.view(N, 1).expand(N, N).t()).float()
        is_neg = target.view(N, 1).expand(N, N).ne(target.view(N, 1).expand(N, N).t()).float()

        if self.hard_mining:
            dist_ap, dist_an = self.hard_example_mining(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = self.weighted_example_mining(dist_mat, is_pos, is_neg)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self.margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float('Inf'):
                loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on

        return loss
