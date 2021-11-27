from typing import List

from torch.utils.data import DataLoader

from datasets import get_augmentation_constructor
from datasets.datasets_loader import ReIDImageDataset


class ReIDTaskPipeline(object):

    def __init__(self, task_list: List, num_workers=0, pin_memory=False):
        self.task_list = task_list
        self.current_task_idx = -1
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.task_round_rest = [task['sustained_round'] for task in task_list]

    def reach_final_task(self) -> bool:
        return self.current_task_idx + 1 == len(self.task_list)

    def get_task(self, idx: int = -1) -> DataLoader:
        task = self.task_list[idx]
        task_name = task['task_name']
        task_dataset_paths = task['dataset_paths']
        img_size = task['img_size']
        norm_mean = task['norm_mean']
        norm_std = task["norm_std"]
        epochs = task['epochs']
        batch_size = task['batch_size']
        augmentation = get_augmentation_constructor(task['augmentation'])(
            size=img_size,
            mean=norm_mean,
            std=norm_std
        )
        none_augmentation = get_augmentation_constructor('none')(
            size=img_size,
            mean=norm_mean,
            std=norm_std
        )

        tr_loader = DataLoader(
            ReIDImageDataset([f'{_path}/train' for _path in task_dataset_paths], augmentation),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        gallery_loader = DataLoader(
            ReIDImageDataset([f'{_path}/gallery' for _path in task["dataset_paths"]], none_augmentation),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        query_loader = DataLoader(
            ReIDImageDataset([f'{_path}/query' for _path in task["dataset_paths"]], none_augmentation),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        return {
            "task_name": task_name,
            "epochs": epochs,
            "tr_loader": tr_loader,
            "query_loader": query_loader,
            "gallery_loaders": gallery_loader
        }

    def current_task(self) -> DataLoader:
        if self.current_task_idx == -1:
            self.current_task_idx = 0
        return self.get_task(self.current_task_idx)

    def next_task(self) -> DataLoader:
        if not self.reach_final_task():
            if self.current_task_idx != -1 and self.task_round_rest[self.current_task_idx]:
                self.task_round_rest[self.current_task_idx] -= 1
            else:
                self.current_task_idx += 1
                self.task_round_rest[self.current_task_idx] -= 1
        return self.current_task()
