from typing import Dict

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from analyse import *
from analysis import marker


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
        # plt.legend(loc='lower right')
        plt.title(f'{client_name}')
        plt.xlabel("Communication Round")
        plt.ylabel(metric_desc)
        plt.savefig(f'{save_path_prefix}_{client_name}_{metric_desc}.svg')


def plot_merged_accuracy_for_many_jobs(
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

    plt.figure(figsize=(4, 4), dpi=300)
    color_id, line_id, marker_id = 0, 0, 0
    for job_name, y_labels in y_labels.items():
        _x, _y = [], []
        for idy, value in enumerate(y_labels):
            if value is not None:
                _x.append(x_labels[idy])
                _y.append(y_labels[idy] / len(client_set) * 100)

        _y = gaussian_filter1d(_y, sigma=0.3)

        plt.plot(_x, _y, linestyle=line_style[line_id], marker=marker[marker_id], label=job_name)

        color_id = (color_id + 1) % len(color)
        line_id = (line_id + 1) % len(line_style)
        marker_id = (marker_id + 1) % len(marker)

    plt.grid(alpha=0.3)
    # plt.legend(loc='lower right', ncol=1)
    plt.xlabel("Communication Round")
    plt.ylabel(metric_desc)
    plt.savefig(f'{save_path_prefix}_{metric_desc}.svg')


if __name__ == '__main__':
    logs_data = load_logs(r'D:\Ryan\Projects\EdgeAI_ReID\logs\2022-3-1\fedstil_k_0-2022-03-05-21-41.json')
    save_path_prefix = r'D:\Ryan\Projects\EdgeAI_ReID\logs\2022-3-1\fedstil_k_0-2022-03-05-21-41'
    accuracy_on_round(
        logs=logs_data['data'],
        rounds=60,
        metric='val_map',
        metric_desc='mAP',
    )
    plot_accuracy_for_one_job(
        logs=logs_data['data'],
        save_path_prefix=save_path_prefix,
        metric='val_map',
        metric_desc='mAP',
    )
    plot_merged_accuracy_for_many_jobs(
        jobs={
            'FedStil Baseline': load_logs(r'../logs/2022-3-1/fedstil_k_9-2022-03-07-21-51.json')['data'],
            'w/o adaptive layer': load_logs(r'../logs/2022-3-1/fedstil-wo-al-2022-03-02-18-17.json')['data'],
            'w/o parameter tying': load_logs(r'../logs/2022-3-1/fedstil-wo-pt-2022-03-03-14-18.json')['data'],
            'w/o s&t knowledge integration': load_logs(r'../logs/2022-3-1/fedstil-wo-st-2022-03-04-10-48.json')['data'],
            # 'FedStil': load_logs(r'../logs/2022-2-23/fedstil_k_7-2022-02-25-03-15.json')['data'],
            # 'FedWeIT': load_logs(r'../logs/2022-2-23/fedweit-2022-02-22-22-24.json')['data'],
            # 'FedCurv': load_logs(r'../logs/2022-2-23/fedcurv-2022-02-20-11-15.json')['data'],
            # 'FedAvg': load_logs(r'../logs/2022-2-23/fedavg-2022-02-20-08-09.json')['data'],
            # 'FedProx': load_logs(r'../logs/2022-3-1/fedprox-2022-03-02-08-10.json')['data'],
            # 'EWC': load_logs(r'../logs/2022-2-23/ewc-2022-02-20-00-15.json')['data'],
            # 'MAS': load_logs(r'../logs/2022-3-1/mas-2022-03-01-23-27.json')['data'],
            # 'iCaRL': load_logs(r'../logs/2022-2-22/icarl-2022-02-11-22-31.json')['data'],
        },
        save_path_prefix=r'D:\Ryan\Projects\EdgeAI_ReID\logs\2022-3-1\abc_ab',
        metric='val_rank_1',
        metric_desc='Rank-1',
    )
