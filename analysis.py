import json
from math import ceil
from typing import List, Dict

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
line_style = ['-', '--', '-.', ':']
marker = ['s', 'o', '^', 'P', '*', 'D', '|', 'v', 'x', '8']


def calculate_accuracy(log_path: str, rounds: int, metric_name: str = 'val_map'):
    # load logs from disk
    with open(log_path, 'r') as f:
        state = json.load(fp=f)

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
        state = json.load(fp=f)

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
        plt_figure: tuple = (30, 4),
        plt_dpi: int = 300,
        col_default_cnt: int = 5,
        y_lim: tuple = (0, 100)
):
    with open(log_path, 'r') as f:
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
        if tasks:
            for client_state in list(state.values()):
                for comm_state in list(client_state.values()):
                    for task_name in list(comm_state.keys()):
                        if task_name not in tasks:
                            del comm_state[task_name]
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
        plt_figure: tuple = (30, 4),
        plt_dpi: int = 300,
        col_default_cnt: int = 5,
        y_lim: tuple = (0, 100)
):
    states = {}
    client_set = set()
    comm_set = set()

    for job_name, job_log_path in job_log_paths.items():
        with open(job_log_path, 'r') as f:
            state = json.load(fp=f)
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
        plt.legend(loc="best")
        plt.title(client_name)
        plt.xlabel("Communication Round")
        plt.ylabel(metric_desc)

    plt.savefig(save_path)


if __name__ == '__main__':
    plot_learning_curve(
        log_path="./logs/2021-11-23/ours-sm_2021-11-25-07-47-24.log",
        save_path="./logs/2021-11-23/ours-sm_2021-11-25-07-47-24_mAP.png",
        metric_name="val_map",
        metric_desc="mAP",
        y_lim=[0, 80]
        # clients=['client-0', 'client-1', 'client-2', 'client-3', 'client-4', ],
        # rounds=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        # tasks=['task-0', 'task-3', 'task-5']
    )
    # calculate_accuracy(
    #     log_path='./logs/2021-11-12/fedcurv_2021-11-15-08-05-58.log',
    #     metric_name="val_map",
    #     rounds=100
    # )
    # calculate_forgetting(
    #     log_path='./logs/2021-11-12/fedcurv_2021-11-15-08-05-58.log',
    #     metric_name="val_map",
    #     rounds=100
    # )
    plot_compared_methods(
        job_log_paths={
            # 'single-model': './logs/2021-11-23/sm_2021-11-23-22-51-28.log',
            # 'multi-model': './logs/2021-11-23/mm_2021-11-24-00-46-27.log',
            # 'ewc': './logs/2021-11-23/ewc_2021-11-23-23-39-29.log',
            # 'mas': './logs/2021-11-23/mas_2021-11-24-02-08-41.log',
            # 'fedavg': './logs/2021-11-23/fedavg_2021-11-24-02-28-05.log',
            # 'fedprox': './logs/2021-11-23/fedprox_2021-11-24-03-53-11.log',
            'fedcurv': './logs/2021-11-23/fedcurv_2021-11-24-01-27-13.log',
            'fedweit': './logs/2021-11-23/fedweit_2021-11-24-00-55-58.log',
            'ours-m': './logs/2021-11-23/ours-mm_2021-11-25-07-45-49.log',
            'ours-s': './logs/2021-11-23/ours-sm_2021-11-25-07-47-24.log'
        },
        save_path='./logs/2021-11-23/total.png',
        metric_name='val_map',
        metric_desc='average mAP',
        clients=['client-0', 'client-1', 'client-2', 'client-3', 'client-4', ],
        rounds=[5, 10, 15, 20, 25, 30, 35, 40],
        y_lim=(0, 80)
    )
