from typing import Callable

from criterions.cross_entropy import CrossEntropyLabelSmooth
from criterions.triplet_loss import TripletLoss

criterion_list = {
    'cross_entropy': CrossEntropyLabelSmooth,
    'triplet_loss': TripletLoss,
}


def get_callable_criterion(criterion_name: str) -> Callable:
    if criterion_name.lower() not in criterion_list.keys():
        raise ValueError(f"Could not find the criterion named '{criterion_name}'.")
    return criterion_list[criterion_name.lower()]
