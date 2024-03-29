from typing import Dict

import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from analyse import *


def accuracy_on_round(
        logs: Dict,
        rounds: int,
        metric: str,
        metric_desc: str
):
    client_avg = []
    for client_name, communication in logs.items():
        task_avg = [
            value[metric] \
            for task_name, value in communication[str(rounds)].items() \
            if metric in value.keys()
        ]
        if len(task_avg):
            task_avg = sum(task_avg) / len(task_avg)
            client_avg.append(task_avg)
            print(f'[{client_name}] {metric} is {task_avg:.2%}')

    client_avg = sum(client_avg) / len(client_avg)
    print(f'Total clients {metric_desc}:{client_avg:.2%}.')


def plot_accuracy_for_one_job(
        logs: Dict,
        save_path_prefix: str,
        metric: str,
        metric_desc: str,
):
    for client_name, communication in logs.items():
        x_labels = [int(v) for v in communication.keys()]
        y_labels = {}  # {task_name: y_labels}

        for comm_id, task_list in communication.items():
            comm_id = int(comm_id)
            for task_name, value in task_list.items():
                if task_name not in y_labels.keys():
                    y_labels[task_name] = [None for _ in x_labels]
                if metric in value.keys():
                    y_labels[task_name][x_labels.index(comm_id)] = value[metric]

        plt.figure(figsize=(4, 4), dpi=300)
        color_id, line_id, marker_id = 0, 0, 0
        for task_name, y_labels in y_labels.items():
            _x, _y = [], []
            for idy, value in enumerate(y_labels):
                if value is not None:
                    _x.append(x_labels[idy])
                    _y.append(y_labels[idy] * 100)

            _y = gaussian_filter1d(_y, sigma=0.1)

            plt.plot(_x, _y, linestyle=line_style[line_id], marker=marker[marker_id], label=task_name)

            plt.grid(alpha=0.3)
            plt.legend(loc='lower right')
            plt.title(f'{client_name}')
            plt.xlabel("Communication Round")
            plt.ylabel(metric_desc)
            plt.savefig(f'{save_path_prefix}_{client_name}_{metric_desc}.svg')

            color_id = (color_id + 1) % len(color)
            line_id = (line_id + 1) % len(line_style)
            marker_id = (marker_id + 1) % len(marker)


def plot_accuracy_for_many_jobs(
        jobs: Dict[str, Dict],
        save_path_prefix: str,
        metric: str,
        metric_desc: str,
):
    logs = {}
    client_set = set()
    comm_set = set()

    for job_name, job_logs in jobs.items():
        logs[job_name] = job_logs
        for client_name in list(job_logs.keys()):
            client_set.add(client_name)
        for client_state in list(job_logs.values()):
            for comm_id in list(client_state.keys()):
                comm_set.add(int(comm_id))

    client_set = sorted(client_set)
    comm_set = sorted(comm_set)

    for client_name in client_set:

        x_labels = [int(v) for v in comm_set]
        y_labels = {}  # { job_name : [ y ] }

        for job_name, job_logs in logs.items():
            y_labels[job_name] = [None for _ in x_labels]
            for comm_id, task_state in job_logs[client_name].items():
                task_avg = [
                    value[metric] \
                    for task_name, value in job_logs[client_name][comm_id].items() \
                    if metric in value.keys()
                ]
                if len(task_avg):
                    task_avg = sum(task_avg) / len(task_avg)
                    y_labels[job_name][x_labels.index(int(comm_id))] = task_avg

        plt.figure(figsize=(4, 4), dpi=300)
        color_id, line_id, marker_id = 0, 0, 0
        for job_name, y_labels in y_labels.items():
            _x, _y = [], []
            for idy, value in enumerate(y_labels):
                if value is not None:
                    _x.append(x_labels[idy])
                    _y.append(y_labels[idy] * 100)

            _y = gaussian_filter1d(_y, sigma=0.1)

            plt.plot(_x, _y, linestyle=line_style[line_id], marker=marker[marker_id], label=job_name)

            color_id = (color_id + 1) % len(color)
            line_id = (line_id + 1) % len(line_style)
            marker_id = (marker_id + 1) % len(marker)

        plt.grid(alpha=0.3)
        plt.legend(loc='lower right')
        plt.title(f'{client_name}')
        plt.xlabel("Communication Round")
        plt.ylabel(metric_desc)
        plt.savefig(f'{save_path_prefix}_{client_name}_{metric_desc}.svg')


def plot_task_accuracy_for_many_jobs(
        jobs: Dict[str, Dict],
        save_path_prefix: str,
        tasks: Dict,
        rounds: list,
        metric: str,
        metric_desc: str,
):
    logs = {}
    client_set = set()
    comm_set = set()

    for job_name, job_logs in jobs.items():
        logs[job_name] = job_logs
        for client_name in list(job_logs.keys()):
            client_set.add(client_name)
        for client_state in list(job_logs.values()):
            for comm_id in list(client_state.keys()):
                comm_set.add(int(comm_id))

    client_set = sorted(client_set)
    comm_set = sorted(comm_set)

    plt.figure(figsize=(12, 3), dpi=300)

    for i, (task_name, task_ids) in enumerate(tasks.items(), 1):
        plt.subplot(1, len(tasks), i)
        plt.figure(1)

        x_labels = [int(v) for v in comm_set]
        y_labels = {}  # { job_name : [ y ] }

        for client_name in client_set:
            for job_name, job_logs in logs.items():
                if job_name not in y_labels.keys():
                    y_labels[job_name] = [None for _ in x_labels]
                for comm_id, task_state in job_logs[client_name].items():
                    task_avg = [
                        _value[metric] \
                        for _task_name, _value in job_logs[client_name][comm_id].items() \
                        if metric in _value.keys() and _task_name in task_ids
                    ]
                    if len(task_avg):
                        task_avg = sum(task_avg) / len(task_avg)
                        if y_labels[job_name][x_labels.index(int(comm_id))] is None:
                            y_labels[job_name][x_labels.index(int(comm_id))] = 0.0
                        y_labels[job_name][x_labels.index(int(comm_id))] += task_avg

        color_id, line_id, marker_id = 0, 0, 0
        for job_name, y_labels in y_labels.items():
            _x, _y = [], []
            for idy, value in enumerate(y_labels):
                if value is not None:
                    _x.append(x_labels[idy])
                    _y.append(y_labels[idy] / len(client_set) * 100)

            print(job_name.lower(), '=', _y)

            _y = gaussian_filter1d(_y, sigma=0.8)

            plt.plot(_x, _y, color=color[color_id],  # linestyle=line_style[line_id],
                     marker=marker[marker_id], label=job_name, linewidth=3)
            plt.title(task_name, fontsize=16)
            color_id = (color_id + 1) % len(color)
            line_id = (line_id + 1) % len(line_style)
            marker_id = (marker_id + 1) % len(marker)

        plt.grid(alpha=0.3)
        plt.xlabel("Communication Round", fontsize=14)
        plt.ylabel(f'{metric_desc} Accuracy', fontsize=14)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
        plt.xlim((rounds[i - 1], 60))
        plt.ylim((40, 80))

    plt.legend(loc='lower right', ncol=1, fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}.pdf')


def plot_merged_accuracy_for_many_jobs(
        jobs: Dict[str, Dict],
        save_path_prefix: str,
):
    logs = {}
    client_set = set()
    comm_set = set()

    for job_name, job_logs in jobs.items():
        logs[job_name] = job_logs
        for client_name in list(job_logs.keys()):
            client_set.add(client_name)
        for client_state in list(job_logs.values()):
            for comm_id in list(client_state.keys()):
                comm_set.add(int(comm_id))

    client_set = sorted(client_set)
    comm_set = sorted(comm_set)

    plt.figure(figsize=(9, 4), dpi=300)
    plt_metrics = [
        ('val_rank_1', 'Rank-1'),
        ('val_map', 'mAP')
    ]

    for i, (metric, metric_desc) in enumerate(plt_metrics, 1):
        plt.subplot(1, 2, i)
        plt.figure(1)

        x_labels = [int(v) for v in comm_set]
        y_labels = {}  # { job_name : [ y ] }

        for client_name in client_set:
            for job_name, job_logs in logs.items():
                if job_name not in y_labels.keys():
                    y_labels[job_name] = [None for _ in x_labels]
                for comm_id, task_state in job_logs[client_name].items():
                    task_avg = [
                        value[metric] \
                        for task_name, value in job_logs[client_name][comm_id].items() \
                        if metric in value.keys()
                    ]
                    if len(task_avg):
                        task_avg = sum(task_avg) / len(task_avg)
                        if y_labels[job_name][x_labels.index(int(comm_id))] is None:
                            y_labels[job_name][x_labels.index(int(comm_id))] = 0.0
                        y_labels[job_name][x_labels.index(int(comm_id))] += task_avg

        color_id, line_id, marker_id = 0, 0, 0
        for job_name, y_labels in y_labels.items():
            _x, _y = [], []
            for idy, value in enumerate(y_labels):
                if value is not None:
                    _x.append(x_labels[idy])
                    _y.append(y_labels[idy] / len(client_set) * 100)

            print(job_name.lower(), '=', _y)

            _y = gaussian_filter1d(_y, sigma=0.1)

            plt.plot(_x, _y, color=color[color_id], linestyle=line_style[line_id],
                     marker=marker[marker_id], label=job_name, linewidth=3)

            color_id = (color_id + 1) % len(color)
            line_id = (line_id + 1) % len(line_style)
            marker_id = (marker_id + 1) % len(marker)

        plt.grid(alpha=0.3)
        plt.xlabel("Communication Round", fontsize=12)
        plt.ylabel(f'{metric_desc} Accuracy', fontsize=12)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
        plt.xlim((0, 60))
        plt.ylim((15, 70))

    plt.legend(loc='lower right', ncol=2, fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}.pdf')


if __name__ == '__main__':
    # accuracy_on_round(
    #     logs=load_logs(r'RESULT_PATH'),
    #     rounds=60,
    #     metric='val_map',  # val_rank_1, val_rank_3, val_rank_5, val_rank_10, val_map
    #     metric_desc='mAP',  # Rank-1, Rank-3, Rank-5, Rank-10, mAP
    # )
    #
    # plot_accuracy_for_one_job(
    #     logs=load_logs(r'RESULT_PATH'),
    #     save_path_prefix=r'SAVE_PREFIX',
    #     metric='val_map',  # val_rank_1, val_rank_3, val_rank_5, val_rank_10, val_map
    #     metric_desc='mAP',  # Rank-1, Rank-3, Rank-5, Rank-10, mAP
    # )
    #
    # plot_task_accuracy_for_many_jobs(
    #     jobs={
    #         'FedSTIL (ours)': load_logs(r'RESULT_PATH')['data'],
    #         'FedWeIT': load_logs(r'RESULT_PATH')['data'],
    #         'FedCurv': load_logs(r'RESULT_PATH')['data'],
    #         'FedAvg': load_logs(r'RESULT_PATH')['data'],
    #         'iCaRL': load_logs(r'RESULT_PATH')['data'],
    #         'EWC': load_logs(r'RESULT_PATH')['data'],
    #     },
    #     save_path_prefix=r'SAVE_PREFIX',
    #     tasks={
    #         'Task-1': ['task-0-0', 'task-1-0', 'task-2-0', 'task-3-0', 'task-4-0'],
    #         'Task-3': ['task-0-2', 'task-1-2', 'task-2-2', 'task-3-2', 'task-4-2'],
    #         'Task-5': ['task-0-4', 'task-1-4', 'task-2-4', 'task-3-4', 'task-4-4'],
    #     },
    #     rounds=[10, 30, 50],
    #     metric='val_map',  # val_rank_1, val_rank_3, val_rank_5, val_rank_10, val_map
    #     metric_desc='mAP',  # Rank-1, Rank-3, Rank-5, Rank-10, mAP
    # )
    #
    # plot_merged_accuracy_for_many_jobs(
    #     jobs={
    #         'FedSTIL (ours)': load_logs(r'RESULT_PATH')['data'],
    #         'FedWeIT': load_logs(r'RESULT_PATH')['data'],
    #         'FedCurv': load_logs(r'RESULT_PATH')['data'],
    #         'FedAvg': load_logs(r'RESULT_PATH')['data'],
    #         'iCaRL': load_logs(r'RESULT_PATH')['data'],
    #         'EWC': load_logs(r'RESULT_PATH')['data'],
    #     },
    #     save_path_prefix=r'SAVE_PREFIX',
    # )

    pass
