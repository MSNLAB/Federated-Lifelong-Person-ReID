from typing import Callable, Dict, Any

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from datasets import augmentations


class ReIDImageDataset(Dataset):

    def __init__(self, root: Dict, transform: Callable = augmentations['none']()):
        super(ReIDImageDataset, self).__init__()
        self.dataset = ImageFolder(root, transform)

    def __getitem__(self, index) -> Any:
        data, class_index = self.dataset[index]
        target = self.dataset.classes[class_index]
        return data, int(target), class_index

    def __len__(self):
        return len(self.dataset)
