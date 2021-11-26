import bisect
from typing import Callable, Dict, Any

from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from torchvision.datasets import ImageFolder

from datasets import augmentation_list


class ReIDImageDataset(Dataset):

    def __init__(self, roots: Dict, transform: Callable = augmentation_list['none']()):
        super(ReIDImageDataset, self).__init__()
        self.dataset = ConcatDataset(datasets=[ImageFolder(root, transform) for root in roots])

    def __getitem__(self, index) -> Any:
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        dataset_idx = bisect.bisect_right(self.dataset.cumulative_sizes, index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.dataset.cumulative_sizes[dataset_idx - 1]
        data, class_index = self.dataset.datasets[dataset_idx][sample_idx]
        target = self.dataset.datasets[dataset_idx].classes[class_index]
        return data, int(target), class_index

    def __len__(self):
        return len(self.dataset)
