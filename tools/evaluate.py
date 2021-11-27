#############################################################################
# Code from : https://gist.github.com/budui/ba3b2c5868f7d68982191be7db32b453
#############################################################################

from typing import Any, Tuple

import numpy as np
import torch


@torch.no_grad()
def get_right_and_junk_index(
        query_label: torch.Tensor,
        gallery_labels: torch.Tensor,
        query_camera_label: torch.Tensor = None,
        gallery_camera_labels: torch.Tensor = None
) -> Tuple[Any, Any]:
    same_label_index = np.argwhere(gallery_labels == query_label)
    if (query_camera_label is not None) and (gallery_camera_labels is not None):
        same_camera_label_index = np.argwhere(gallery_camera_labels == query_camera_label)
        # the index of mis-detected images, which contain the body parts.
        junk_index1 = np.argwhere(gallery_labels == -1)
        # find index that are both in query_index and camera_index
        # the index of the images, which are of the same identity in the same cameras.
        junk_index2 = np.intersect1d(same_label_index, same_camera_label_index)
        junk_index = np.append(junk_index2, junk_index1)

        # find index that in query_index but not in camera_index
        # which means the same label but different camera
        right_index = np.setdiff1d(same_label_index, same_camera_label_index, assume_unique=True)
        return right_index, junk_index
    else:
        return same_label_index, None


@torch.no_grad()
def evaluate_with_index(
        sorted_similarity_index: torch.Tensor,
        right_result_index: torch.Tensor,
        junk_result_index: torch.Tensor = None
) -> Tuple[Any, Any]:
    """calculate cmc curve and Average Precision for a single query with index
    :param sorted_similarity_index: index of all returned items. typically get with
        function `np.argsort(similarity)`
    :param right_result_index: index of right items. such as items in gallery
        that have the same id but different camera with query
    :param junk_result_index: index of junk items. such as items in gallery
        that have the same camera and id with query
    :return: single cmc, Average Precision
    """
    # initial a numpy array to store the AccK(like [0, 0, 0, 1, 1, ...,1]).
    cmc = np.zeros(len(sorted_similarity_index), dtype=np.int32)
    ap = 0.0

    if len(right_result_index[0]) == 0:
        cmc[0] = -1
        return cmc, ap
    if junk_result_index is not None:
        # remove junk_index
        # all junk_result_index in sorted_similarity_index has been removed.
        # for example:
        # (sorted_similarity_index, junk_result_index)
        # ([3, 2, 0, 1, 4],         [0, 1])             -> [3, 2, 4]
        need_remove_mask = np.in1d(sorted_similarity_index, junk_result_index, invert=True)
        sorted_similarity_index = sorted_similarity_index[need_remove_mask]

    mask = np.in1d(sorted_similarity_index, right_result_index)
    right_index_location = np.argwhere(mask == True).flatten()

    # [0,0,0,...0, 1,1,1,...,1]
    #              |
    #  right answer first appearance
    cmc[right_index_location[0]:] = 1

    for i in range(len(right_result_index)):
        precision = float(i + 1) / (right_index_location[i] + 1)
        if right_index_location[i] != 0:
            # last rank precision, not last match precision
            old_precision = float(i) / (right_index_location[i])
        else:
            old_precision = 1.0
        ap = ap + (1.0 / len(right_result_index)) * (old_precision + precision) / 2

    return cmc, ap


@torch.no_grad()
def calculate_similarity_distance(
        query_feature: torch.Tensor,
        gallery_features: torch.Tensor
) -> Any:
    """calculate the distance between query and gallery
    :param gallery_features: the feature's list for gallery
    :param query_feature: the feature for query
    :return: similarity_distance, size = N*1
    """
    if isinstance(query_feature, np.ndarray):
        return np.dot(gallery_features, query_feature)
    else:
        return torch.mm(gallery_features, query_feature.view(-1, 1)).squeeze(1).cpu().numpy()


@torch.no_grad()
def evaluate(
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        gallery_features: torch.Tensor,
        gallery_labels: torch.Tensor,
        query_camera_labels: torch.Tensor = None,
        gallery_camera_labels: torch.Tensor = None,
        device: str = 'cpu'
) -> Tuple[Any, Any]:
    # copy the tensor in device to host memory first
    query_features = query_features.to(device)
    gallery_features = gallery_features.to(device)
    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()
    if query_camera_labels is not None:
        query_camera_labels = query_camera_labels.cpu()
    if gallery_camera_labels is not None:
        gallery_camera_labels = gallery_camera_labels.cpu()

    total_cmc = np.zeros(len(gallery_labels), dtype=np.int32)
    total_average_precision = 0.0

    for i in range(len(query_labels)):
        similarity_distance = calculate_similarity_distance(query_features[i], gallery_features)
        cmc, ap = evaluate_with_index(
            np.argsort(similarity_distance)[::-1],
            *get_right_and_junk_index(
                query_labels[i], gallery_labels,
                query_camera_labels[i] if query_camera_labels is not None else None,
                gallery_camera_labels if gallery_camera_labels is not None else None
            )
        )

        if cmc[0] == -1:
            continue
        total_cmc += cmc
        total_average_precision += ap

    return total_cmc.astype(np.float64) / len(query_labels), total_average_precision / len(query_labels)
