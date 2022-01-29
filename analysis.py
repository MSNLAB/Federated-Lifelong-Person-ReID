import json
import math
from math import ceil
from typing import List, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter1d

color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
line_style = ['-', '--', '-.', ':']
marker = ['s', 'o', '^', 'P', '*', 'D', '|', 'v', 'x', '8']


def calculate_accuracy(log_path: str, rounds: int, metric_name: str = 'val_map'):
    # load logs from disk
    with open(log_path, 'r') as f:
        state = json.load(fp=f)['data']

    client_avg = []  # { client_name: avg mAP }
    for client_name, communication in state.items():
        task_avg = []
        for task_name, value in communication[str(rounds)].items():
            task_avg.append(value[metric_name])

        task_avg = sum(task_avg) / len(task_avg)
        client_avg.append(task_avg)
        print(f'[{client_name}] {metric_name} is {task_avg}')

    client_avg = sum(client_avg) / len(client_avg)
    print(f'Total clients {metric_name}:{client_avg:.2%}.')


def calculate_forgetting(log_path: str, rounds: int, metric_name: str = 'val_map'):
    # load logs from disk
    with open(log_path, 'r') as f:
        state = json.load(fp=f)['data']

    client_forgetting = []  # { client_name: avg mAP }
    for client_name, communication in state.items():
        task_forgetting = []
        for task_name, value in communication[str(rounds)].items():
            max_value = max([
                _value[metric_name] \
                for comm, task_list in communication.items() \
                for _task_name, _value in task_list.items() \
                if _task_name == task_name
                   and int(comm) <= rounds
                   and metric_name in _value.keys()
            ])
            task_forgetting.append(max_value - value[metric_name])

        task_forgetting = sum(task_forgetting) / len(task_forgetting)
        client_forgetting.append(task_forgetting)
        print(f'[{client_name}] {metric_name} forgetting is {task_forgetting:.2%}')

    client_avg = sum(client_forgetting) / len(client_forgetting)
    print(f'Total clients {metric_name}:{client_avg:.2%}.')


def plot_learning_curve(
        log_path: str,
        save_path: str,
        metric_name: str = 'val_map',
        metric_desc: str = 'mean Average Precision',
        clients: List[str] = None,
        rounds: List[int] = None,
        tasks: List[str] = None,
        plt_figure: tuple = (20, 3),
        plt_dpi: int = 300,
        col_default_cnt: int = 5,
        y_lim: tuple = (0, 100)
):
    with open(log_path, 'r') as f:
        state = json.load(fp=f)['data']
        if clients:
            for client_name in list(state.keys()):
                if client_name not in clients:
                    del state[client_name]
        if rounds:
            for client_state in list(state.values()):
                for comm in list(client_state.keys()):
                    if int(comm) not in rounds:
                        del client_state[comm]
        if tasks:
            for client_state in list(state.values()):
                for comm_state in list(client_state.values()):
                    for task_name in list(comm_state.keys()):
                        if task_name not in tasks:
                            del comm_state[task_name]

    state = {k: v for k, v in sorted(state.items(), key=lambda v: v[0])}

    client_num = len(state.keys())

    plt.figure(figsize=plt_figure, dpi=plt_dpi)
    plt_rows = ceil(client_num / col_default_cnt)
    plt_cols = client_num if client_num < col_default_cnt else col_default_cnt

    for idx, (client_name, communication) in enumerate(state.items(), 1):
        plt.subplot(plt_rows, plt_cols, idx)

        x_labels = [int(v) for v in communication.keys()]
        y_labels = {}

        for comm, task_list in communication.items():
            for task_name, value in task_list.items():
                if task_name not in y_labels.keys():
                    y_labels[task_name] = [None for _ in x_labels]
                if metric_name in value.keys():
                    y_labels[task_name][x_labels.index(int(comm))] = value[metric_name]

        c_id, l_id, m_id = 0, 0, 0
        for task_name, y_labels in y_labels.items():
            _x_labels = []
            _y_labels = []
            for idy, value in enumerate(y_labels):
                if value:
                    _x_labels.append(x_labels[idy])
                    _y_labels.append(y_labels[idy] * 100)

            _y_labels = gaussian_filter1d(_y_labels, sigma=1.2)

            plt.plot(_x_labels, _y_labels, linestyle=line_style[l_id], marker=marker[m_id], label=task_name)
            c_id = c_id + 1 if c_id + 1 < len(color) else 0
            l_id = l_id + 1 if l_id + 1 < len(line_style) else 0
            m_id = m_id + 1 if m_id + 1 < len(marker) else 0

        plt.ylim(y_lim)
        plt.legend(loc="best")
        plt.title(client_name)
        plt.xlabel("Communication Round")
        plt.ylabel(metric_desc)

    plt.savefig(save_path)


def plot_compared_methods(
        job_log_paths: Dict[str, str],
        save_path: str,
        metric_name: str = 'val_map',
        metric_desc: str = 'mean Average Precision',
        clients: List[str] = None,
        rounds: List[int] = None,
        plt_figure: tuple = (23, 4),
        plt_dpi: int = 300,
        col_default_cnt: int = 5,
        y_lim: tuple = (0, 100)
):
    states = {}
    client_set = set()
    comm_set = set()

    for job_name, job_log_path in job_log_paths.items():
        with open(job_log_path, 'r') as f:
            state = json.load(fp=f)['data']
            if clients:
                for client_name in list(state.keys()):
                    if client_name not in clients:
                        del state[client_name]
                    else:
                        client_set.add(client_name)
            if rounds:
                for client_state in list(state.values()):
                    for comm in list(client_state.keys()):
                        if int(comm) not in rounds:
                            del client_state[comm]
                        else:
                            comm_set.add(int(comm))
            states[job_name] = state
    client_set = sorted(client_set)
    comm_set = sorted(comm_set)

    client_num = len(client_set)
    plt.figure(figsize=plt_figure, dpi=plt_dpi)
    plt_rows = ceil(client_num / col_default_cnt)
    plt_cols = client_num if client_num < col_default_cnt else col_default_cnt

    for idx, client_name in enumerate(client_set, 1):
        plt.subplot(plt_rows, plt_cols, idx)

        x_labels = [int(v) for v in comm_set]
        y_labels = {}  # { job_name : [ y ] }

        for job_name, job_state in states.items():
            y_labels[job_name] = [None for _ in x_labels]
            for comm, task_state in job_state[client_name].items():
                task_avg = []
                for task_name, value in job_state[client_name][comm].items():
                    task_avg.append(value[metric_name])
                task_avg = sum(task_avg) / len(task_avg)
                y_labels[job_name][x_labels.index(int(comm))] = task_avg

        c_id, l_id, m_id = 0, 0, 0
        for job_name, y_labels in y_labels.items():
            _x_labels = []
            _y_labels = []
            for idy, value in enumerate(y_labels):
                if value:
                    _x_labels.append(x_labels[idy])
                    _y_labels.append(y_labels[idy] * 100)

            _y_labels = gaussian_filter1d(_y_labels, sigma=1.2)

            plt.plot(_x_labels, _y_labels, linestyle=line_style[l_id], marker=marker[m_id], label=job_name)
            c_id = c_id + 1 if c_id + 1 < len(color) else 0
            l_id = l_id + 1 if l_id + 1 < len(line_style) else 0
            m_id = m_id + 1 if m_id + 1 < len(marker) else 0

        # plt.ylim(y_lim)
        plt.legend(loc="lower right")
        plt.title(client_name)
        plt.xlabel("Communication Round")
        plt.ylabel(metric_desc)

    plt.savefig(save_path)


def plot_representation(
        job_log_paths: Dict[str, str],
        save_path: str,
        clients: List[str] = None,
        rounds: List[int] = None,
        tasks: List[str] = None,
        plt_figure: tuple = (20, 4),
        plt_dpi: int = 300,
):
    states = {}

    for job_name, job_log_path in job_log_paths.items():
        with open(job_log_path, 'r') as f:
            state = json.load(fp=f)
            if clients:
                for client_name in list(state.keys()):
                    if client_name not in clients:
                        del state[client_name]
            if rounds:
                for client_state in list(state.values()):
                    for comm in list(client_state.keys()):
                        if int(comm) not in rounds:
                            del client_state[comm]
            states[job_name] = state

    plt.figure(figsize=plt_figure, dpi=plt_dpi)
    plt_rows = 1
    plt_cols = len(states)

    for idx, (job_name, job_state) in enumerate(states.items(), 1):
        rep_avg = []
        for comm, task_state in job_state[client_name].items():
            for task_name, value in job_state[client_name][comm].items():
                rep_avg.append(torch.Tensor(value['val_avg_representation']))
        rep_avg = sum(rep_avg) / len(rep_avg)
        plt.subplot(plt_rows, plt_cols, idx)
        size = int(math.sqrt(len(rep_avg)))
        # rep_avg = torch.pow(10, rep_avg * 1)
        # rep_avg = torch.nn.functional.normalize(rep_avg, p=10, dim=0)
        # print(rep_avg)
        rep_avg = rep_avg[0:size * size].reshape(size, size)
        plt.title(job_name)
        plt.imshow(rep_avg, vmin=torch.min(rep_avg), vmax=torch.max(rep_avg), cmap=plt.cm.Blues)
        plt.xticks([])
        plt.yticks([])

    plt.savefig(save_path)


def plot_rep_distribution(
        job_log_paths: Dict[str, str],
        save_path: str,
        clients: List[str] = None,
        rounds: List[int] = None,
        tasks: List[str] = None,
        plt_figure: tuple = (30, 8),
        plt_dpi: int = 300,
):
    states = {}

    for job_name, job_log_path in job_log_paths.items():
        with open(job_log_path, 'r') as f:
            state = json.load(fp=f)
            if clients:
                for client_name in list(state.keys()):
                    if client_name not in clients:
                        del state[client_name]
            if rounds:
                for client_state in list(state.values()):
                    for comm in list(client_state.keys()):
                        if int(comm) not in rounds:
                            del client_state[comm]
            states[job_name] = state

    plt_rows = 1
    plt_cols = len(states)
    plt.figure(figsize=plt_figure, dpi=plt_dpi)

    for idx, (job_name, job_state) in enumerate(states.items(), 1):
        ax = plt.subplot(plt_rows, plt_cols, idx, projection='3d')
        ax.view_init(elev=30., azim=45)

        x_label = []
        y_label = []
        z_label = []
        for comm, task_state in job_state[client_name].items():
            rep_avg = []
            for task_name, value in job_state[client_name][comm].items():
                rep_avg.append(torch.Tensor(value['val_avg_representation']))
            rep_avg = sum(rep_avg) / len(rep_avg)

            _x_label = np.arange(0, 0.05, 0.001)
            _y_label = _x_label * 0 + int(comm)
            _z_label = [0 for _ in range(len(_x_label))]
            for v in rep_avg:
                if int(v / 0.001) < len(_z_label):
                    _z_label[int(v / 0.001)] += 1
            x_label.extend(_x_label)
            y_label.extend(_y_label)
            z_label.extend(_z_label)
            ax.plot3D(_x_label, _y_label, _z_label, linestyle=':', color='cornflowerblue')
            ax.add_collection3d(Poly3DCollection([[
                *[(_x_label[i], _y_label[i], _z_label[i]) for i in range(len(_x_label))],
                *[(min(_x_label), min(_y_label), 0.0)],
            ]], linestyle=':', color='cornflowerblue', alpha=0.35))

        ax.plot_trisurf(x_label, y_label, z_label, cmap='viridis', edgecolor='none', alpha=0.45)
        ax.set_xlabel("Representation Value")
        ax.set_ylabel("Communication Round")
        ax.set_zlabel("Count")
        plt.title(job_name)
    plt.savefig(save_path)


def plot_rep_distributions(
        job_log_paths: Dict[str, str],
        save_path: str,
        clients: List[str] = None,
        rounds: List[int] = None,
        tasks: List[str] = None,
        plt_figure: tuple = (6, 4),
        plt_dpi: int = 300,
):
    states = {}

    for job_name, job_log_path in job_log_paths.items():
        with open(job_log_path, 'r') as f:
            state = json.load(fp=f)
            if clients:
                for client_name in list(state.keys()):
                    if client_name not in clients:
                        del state[client_name]
            if rounds:
                for client_state in list(state.values()):
                    for comm in list(client_state.keys()):
                        if int(comm) not in rounds:
                            del client_state[comm]
            states[job_name] = state

    plt.figure(figsize=plt_figure, dpi=plt_dpi)

    for idx, (job_name, job_state) in enumerate(states.items(), 1):
        rep_avg = []
        for comm, task_state in job_state[client_name].items():
            for task_name, value in job_state[client_name][comm].items():
                rep_avg.append(torch.Tensor(value['val_avg_representation']))
        rep_avg = sum(rep_avg) / len(rep_avg)
        x_label = list(np.arange(0, torch.max(rep_avg), 0.001))
        y_label = [0 for _ in range(len(x_label))]
        for v in rep_avg:
            y_label[int(v / 0.001)] += 1
        y_label = gaussian_filter1d(y_label, sigma=2.2)
        plt.plot(x_label, y_label, label=job_name)
        plt.fill_between(x_label, 0 * y_label, y_label, alpha=0.05)
        plt.xlim(0.0, 0.04)
        plt.ylim(0, 280)
    plt.legend(loc="upper right")
    plt.xlabel("Representation Value")
    plt.ylabel("Counts")
    plt.title("Representation Distribution")
    # plt.grid(alpha=0.6)
    plt.savefig(save_path)


if __name__ == '__main__':
    plot_learning_curve(
        log_path="./logs/2022-1-30/fedweit-2022-01-24-00-33.json",
        save_path="./logs/2022-1-30/fedweit-2022-01-24-00-33_mAP.png",
        metric_name="val_map",
        metric_desc="mAP",
        y_lim=[0, 90],
        # clients=['client-0', 'client-1', 'client-2', 'client-3', 'client-4', ],
        # rounds=[10, 30, 50, 70,  90],
        # tasks=['task-1', 'task-2', 'task-3']
    )
    # plot_compared_methods(
    #     job_log_paths={
    #         'single-model': './logs/2022-1-18/sm-2022-01-17-13-41.json',
    #         'multi-model': './logs/2022-1-18/mm-2022-01-17-15-18.json',
    #         'ewc': './logs/2022-1-18/ewc-2022-01-17-16-56.json',
    #         'mas': './logs/2022-1-18/mas-2022-01-17-19-12.json',
    #         'fedavg': './logs/2022-1-18/fedavg-2022-01-18-17-38.json',
    #         'fedprox': './logs/2022-1-18/fedprox-2022-01-18-19-34.json',
    #         'fedcurv': './logs/2022-1-18/fedcurv-2022-01-18-22-45.json',
    #         'fedweit': './logs/2022-1-18/fedweit-2022-01-19-17-30.json',
    #         'fedstil': './logs/2022-1-18/fedstil-2022-01-19-12-55.json',
    #     },
    #     save_path='./logs/2022-1-18/total.png',
    #     metric_name='val_map',
    #     metric_desc='average mAP',
    #     clients=['client-0', 'client-1', 'client-2', 'client-3', 'client-4', ],
    #     rounds=[10, 20, 30, 40, 50, 60, 70, 80],
    #     y_lim=(30, 82)
    # )
    calculate_accuracy(
        log_path='./logs/2022-1-30/fedweit-2022-01-24-00-33.json',
        metric_name="val_map",
        rounds=80
    )
    calculate_forgetting(
        log_path='./logs/2022-1-30/fedweit-2022-01-24-00-33.json',
        metric_name="val_map",
        rounds=80
    )
    # plot_representation(
    #     job_log_paths={
    #         'baseline': './logs/2021-12-6/sm_2021-12-06-05-39-40.log',
    #         # 'multi-model': './logs/2021-12-6/mm_2021-12-06-11-43-16.log',
    #         # 'ewc': './logs/2021-12-6/ewc_2021-12-06-19-36-33.log',
    #         # 'mas': './logs/2021-12-6/mas_2021-12-07-04-25-16.log',
    #         'fedavg': './logs/2021-12-6/fedavg_2021-12-07-10-51-06.log',
    #         # 'fedprox': './logs/2021-12-6/fedprox_2021-12-07-16-39-03.log',
    #         'fedweit': './logs/2021-12-6/fedweit_2021-12-06-12-44-28.log',
    #         'fedcurv': './logs/2021-12-6/fedcurv_2021-12-07-09-33-19.log',
    #         'fedreil(ours-m)': './logs/2021-12-6/ours-mm_2021-12-07-12-00-36.log',
    #         'fedreil(ours-s)': './logs/2021-12-6/ours-sm_2021-12-07-11-57-00.log'
    #     },
    #     save_path='./logs/2021-12-6/rep_total.png',
    #     clients=['client-0', 'client-1', 'client-2', 'client-3', 'client-4'],
    #     rounds=[80],
    # )
    # plot_rep_distribution(
    #     job_log_paths={
    #         'baseline': './logs/2021-12-6/sm_2021-12-06-05-39-40.log',
    #         # 'multi-model': './logs/2021-12-6/mm_2021-12-06-11-43-16.log',
    #         'ewc': './logs/2021-12-6/ewc_2021-12-06-19-36-33.log',
    #         # 'mas': './logs/2021-12-6/mas_2021-12-07-04-25-16.log',
    #         'fedavg': './logs/2021-12-6/fedavg_2021-12-07-10-51-06.log',
    #         # 'fedprox': './logs/2021-11-30/fedprox_2021-11-30-04-14-53.log',
    #         'fedweit': './logs/2021-12-6/fedweit_2021-12-06-12-44-28.log',
    #         'fedcurv': './logs/2021-12-6/fedcurv_2021-12-07-09-33-19.log',
    #         # 'fedreil(ours-m)': './logs/2021-12-6/ours-mm_2021-12-07-12-00-36.log',
    #         'fedreil(ours-s)': './logs/2021-12-6/ours-sm_2021-12-07-11-57-00.log'
    #     },
    #     save_path='./logs/2021-12-6/rep_dis_change_total.png',
    #     clients=['client-0', 'client-1', 'client-2', 'client-3', 'client-4'],
    #     rounds=[10, 20, 30, 40, 50, 60, 70, 80],
    # )
    # plot_rep_distributions(
    #     job_log_paths={
    #         'baseline': './logs/2021-12-6/sm_2021-12-06-05-39-40.log',
    #         # 'multi-model': './logs/2021-12-6/mm_2021-12-06-11-43-16.log',
    #         # 'ewc': './logs/2021-12-6/ewc_2021-12-06-19-36-33.log',
    #         # 'mas': './logs/2021-12-6/mas_2021-12-07-04-25-16.log',
    #         'fedavg': './logs/2021-12-6/fedavg_2021-12-07-10-51-06.log',
    #         # 'fedprox': './logs/2021-11-30/fedprox_2021-11-30-04-14-53.log',
    #         'fedweit': './logs/2021-12-6/fedweit_2021-12-06-12-44-28.log',
    #         'fedcurv': './logs/2021-12-6/fedcurv_2021-12-07-09-33-19.log',
    #     },
    #     save_path='./logs/2021-12-6/rep_dis_total.png',
    #     clients=['client-0', 'client-1', 'client-2', 'client-3', 'client-4'],
    #     rounds=[80],
    # )
