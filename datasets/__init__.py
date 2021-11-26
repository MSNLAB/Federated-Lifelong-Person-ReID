from typing import Callable

from datasets.image_augmentation import *

augmentation_list = {
    'none': augmentation_none,
    'default': augmentation_default,
    'rose': augmentation_rose,
    'sharp': augmentation_sharp,
    'drastic': augmentation_drastic,
}


def get_callable_augmentation(augmentation_name: str) -> Callable:
    if augmentation_name.lower() not in augmentation_list.keys():
        raise ValueError(f"Could not find the image augmentation named '{augmentation_name}'.")
    return augmentation_list[augmentation_name.lower()]
