import os
from typing import List, Dict

from torch.utils.data import DataLoader

from datasets import augmentations
from datasets.datasets_loader import ReIDImageDataset


class ReIDTaskPipeline(object):

    def __init__(self, task_list: List, task_opts: Dict, datasets_dir: str):
        self.task_list = task_list
        self.task_opts = task_opts
        self.datasets_dir = datasets_dir
        self.current_task_idx = -1
        self.task_round_rest = [task_opts['sustain_rounds'] for _ in task_list]

    def reach_final_task(self) -> bool:
        return self.current_task_idx + 1 == len(self.task_list)

    def get_task(self, idx: int = -1) -> DataLoader:
        task = self.task_list[idx]
        task_paths = os.path.join(self.datasets_dir, task)

        tr_augmentation = augmentations[self.task_opts['augment_opts']['level']](
            size=self.task_opts['augment_opts']['img_size'],
            mean=self.task_opts['augment_opts']['norm_mean'],
            std=self.task_opts['augment_opts']['norm_std']
        )
        none_augmentation = augmentations['none'](
            size=self.task_opts['augment_opts']['img_size'],
            mean=self.task_opts['augment_opts']['norm_mean'],
            std=self.task_opts['augment_opts']['norm_std']
        )

        tr_dataset = ReIDImageDataset(os.path.join(task_paths, 'train'), tr_augmentation)
        tr_loader = DataLoader(
            dataset=tr_dataset,
            shuffle=True,
            drop_last=len(tr_dataset) % self.task_opts['loader_opts']['batch_size'] == 1,
            batch_size=self.task_opts['loader_opts']['batch_size'],
            num_workers=self.task_opts['loader_opts']['num_workers'],
            pin_memory=self.task_opts['loader_opts']['pin_memory'],
            persistent_workers=self.task_opts['loader_opts']['persistent_workers'],
            multiprocessing_context=self.task_opts['loader_opts']['multiprocessing_context'],
        )

        query_dataset = ReIDImageDataset(os.path.join(task_paths, 'query'), none_augmentation)
        query_loader = DataLoader(
            dataset=query_dataset,
            shuffle=False,
            drop_last=len(query_dataset) % self.task_opts['loader_opts']['batch_size'] == 1,
            batch_size=self.task_opts['loader_opts']['batch_size'],
            num_workers=self.task_opts['loader_opts']['num_workers'],
            pin_memory=self.task_opts['loader_opts']['pin_memory'],
            persistent_workers=self.task_opts['loader_opts']['persistent_workers'],
            multiprocessing_context=self.task_opts['loader_opts']['multiprocessing_context'],
        )

        gallery_dataset = ReIDImageDataset(os.path.join(task_paths, 'gallery'), none_augmentation)
        gallery_loader = DataLoader(
            dataset=gallery_dataset,
            shuffle=False,
            drop_last=len(gallery_dataset) % self.task_opts['loader_opts']['batch_size'] == 1,
            batch_size=self.task_opts['loader_opts']['batch_size'],
            num_workers=self.task_opts['loader_opts']['num_workers'],
            pin_memory=self.task_opts['loader_opts']['pin_memory'],
            persistent_workers=self.task_opts['loader_opts']['persistent_workers'],
            multiprocessing_context=self.task_opts['loader_opts']['multiprocessing_context'],
        )

        return {
            "task_name": task,
            "tr_epochs": self.task_opts['train_epochs'],
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
