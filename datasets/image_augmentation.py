from typing import Tuple

import torchvision.transforms as T


def augmentation_none(
        size: Tuple = (384, 128),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
        T.Resize(size),
    ])


def augmentation_default(
        size: Tuple = (384, 128),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomErasing(p=0.5),
        T.Resize(size),
    ])


def augmentation_rose(
        size: Tuple = (384, 128),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomErasing(p=0.6),
        T.Resize(size),
    ])


def augmentation_sharp(
        size: Tuple = (384, 128),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomErasing(p=0.75),
        T.Resize(size),
    ])


def augmentation_drastic(
        size: Tuple = (384, 128),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomErasing(p=0.9),
        T.Resize(size),
    ])
