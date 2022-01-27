from typing import Callable, Dict, Any, Union

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from datasets import augmentations


class ReIDImageDataset(Dataset):

    def __init__(self, source: Union[str, Dict], transform: Callable = augmentations['none']()):
        super(ReIDImageDataset, self).__init__()

        if isinstance(source, str):
            self.dataset = ImageFolder(source, transform)
            self.classes = [int(class_idx) for class_idx in self.dataset.classes]
        elif isinstance(source, dict):
            self.dataset = [(img, index) for class_idx, imgs in source.items() for img, index in imgs]
            self.classes = [int(class_idx) for class_idx, imgs in source.items() for _ in imgs]
        else:
            raise ValueError("Input source should be path in disk or dictionary in memory.")

    @property
    def person_ids(self):
        return self.classes

    def __getitem__(self, index) -> Any:
        data, class_index = self.dataset[index]
        person_id = self.classes[class_index]
        return data, int(person_id), class_index

    def __len__(self):
        return len(self.dataset)
