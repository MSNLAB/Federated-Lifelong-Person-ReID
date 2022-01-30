from typing import Callable, Dict, Any, Union

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from datasets import augmentations


class ReIDImageDataset(Dataset):

    def __init__(self, source: Union[str, Dict], transform: Callable = augmentations['none']()):
        super(ReIDImageDataset, self).__init__()
        self.reload_source(source, transform)

    def reload_source(self, source, transform: Callable = augmentations['none']()):
        if isinstance(source, str):
            self.dataset = ImageFolder(source, transform)
            self.classes = [int(class_idx) for class_idx in self.dataset.classes]
        elif isinstance(source, dict):
            self.dataset = []
            self.classes = {}
            for person_id, protos in source.items():
                for img, class_id in protos:
                    self.dataset.append((img, class_id))
                    self.classes[class_id] = person_id
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
