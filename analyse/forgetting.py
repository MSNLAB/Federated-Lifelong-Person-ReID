from typing import Dict

from matplotlib import pyplot as plt

from analyse import *


def forgetting_on_round(
        logs: Dict,
        rounds: int,
        metric: str,
        metric_desc: str
):
    client_forget = []
    for client_name, communication in logs.items():
        highest_metric_value = {}
        for _round, metric_values in communication.items():
            _round = int(_round)
            if _round > rounds:
                break
            for task_name, values in metric_values.items():
                if metric in values.keys():
                    if task_name not in highest_metric_value.keys():
                        highest_metric_value[task_name] = (values[metric], _round)
                    elif values[metric] > highest_metric_value[task_name][0]:
                        highest_metric_value[task_name] = (values[metric], _round)

        task_forget = []
        for task_name, (value, _round) in highest_metric_value.items():
            for _sr in range(_round + 1, rounds + 1):
                _sr = str(_sr)
                if task_name in communication[_sr].keys() and metric in communication[_sr][task_name].keys():
                    task_forget.append(value - communication[_sr][task_name][metric])

        if len(task_forget):
            task_forget = sum(task_forget) / len(task_forget)
            client_forget.append(task_forget)
            print(f'[{client_name}] {metric} has forgetting {task_forget:.2%}')

    client_forget = sum(client_forget) / len(client_forget)
    print(f'Total clients {metric_desc} has forgetting {client_forget:.2%}.')


def plot_forgetting_for_many_jobs(
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
        data = {}  # { job_name : [ y ] }
        for job_name, job_logs in logs.items():
            data[job_name] = 0

            highest_metric_value = {}
            for _round, metric_values in job_logs[client_name].items():
                _round = int(_round)
                for task_name, values in metric_values.items():
                    if metric in values.keys():
                        if task_name not in highest_metric_value.keys():
                            highest_metric_value[task_name] = (values[metric], _round)
                        elif values[metric] > highest_metric_value[task_name][0]:
                            highest_metric_value[task_name] = (values[metric], _round)

            task_forget = []
            for task_name, (value, _round) in highest_metric_value.items():
                for _sr in range(_round + 1, comm_set[-1] + 1):
                    _sr = str(_sr)
                    if task_name in job_logs[client_name][_sr].keys() and \
                            metric in job_logs[client_name][_sr][task_name].keys():
                        task_forget.append(value - job_logs[client_name][_sr][task_name][metric])

            if len(task_forget):
                task_forget = sum(task_forget) / len(task_forget)
                data[job_name] = task_forget

        plt.figure(figsize=(5, 5), dpi=300)
        plt.bar(range(len(data)), data.values(), tick_label=list(data.keys()))
        plt.xticks(rotation=45)

        plt.title(f'{client_name}')
        plt.xlabel("Rehearsal Size")
        plt.ylabel(metric_desc)
        plt.savefig(f'{save_path_prefix}_{client_name}_{metric_desc}.svg')


def plot_merged_forgetting_for_many_jobs(
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

    forget_data = {}  # { job_name : [ y ] }

    for client_name in client_set:
        for job_name, job_logs in logs.items():
            if job_name not in forget_data.keys():
                forget_data[job_name] = 0

            highest_metric_value = {}
            for _round, metric_values in job_logs[client_name].items():
                _round = int(_round)
                for task_name, values in metric_values.items():
                    if metric in values.keys():
                        if task_name not in highest_metric_value.keys():
                            highest_metric_value[task_name] = (values[metric], _round)
                        elif values[metric] > highest_metric_value[task_name][0]:
                            highest_metric_value[task_name] = (values[metric], _round)

            task_forget = []
            for task_name, (value, _round) in highest_metric_value.items():
                for _sr in range(_round + 1, comm_set[-1] + 1):
                    _sr = str(_sr)
                    if task_name in job_logs[client_name][_sr].keys() and \
                            metric in job_logs[client_name][_sr][task_name].keys():
                        task_forget.append(value - job_logs[client_name][_sr][task_name][metric])

            if len(task_forget):
                task_forget = sum(task_forget) / len(task_forget)
                forget_data[job_name] += task_forget / len(client_set)

        plt.figure(figsize=(6, 6), dpi=300)
        plt.bar(range(len(forget_data)), forget_data.values(), tick_label=list(forget_data.keys()))
        plt.xticks(rotation=45)
        plt.xlabel("Rehearsal Size")
        plt.ylabel(metric_desc)
        plt.savefig(f'{save_path_prefix}_{metric_desc}.svg')


if __name__ == '__main__':
    logs_data = load_logs(r'D:\Ryan\Projects\EdgeAI_ReID\logs\2022-3-1\fedstil_k_0-2022-03-05-21-41.json')
    forgetting_on_round(
        logs=logs_data['data'],
        rounds=60,
        metric='val_map',
        metric_desc='mAP',
    )
    plot_merged_forgetting_for_many_jobs(
        jobs={
            '0K': load_logs(r'../logs/2022-3-1/fedstil_k_0-2022-03-05-21-41.json')['data'],
            '2K': load_logs(r'../logs/2022-3-1/fedstil_k_1-2022-03-07-00-05.json')['data'],
            '4K': load_logs(r'../logs/2022-3-1/fedstil_k_2-2022-03-05-23-32.json')['data'],
            '6K': load_logs(r'../logs/2022-3-1/fedstil_k_3-2022-03-07-04-20.json')['data'],
            '8K': load_logs(r'../logs/2022-3-1/fedstil_k_4-2022-03-06-02-25.json')['data'],
            '10K': load_logs(r'../logs/2022-3-1/fedstil_k_5-2022-03-07-09-21.json')['data'],
            '12K': load_logs(r'../logs/2022-3-1/fedstil_k_6-2022-03-06-06-13.json')['data'],
            '14K': load_logs(r'../logs/2022-3-1/fedstil_k_7-2022-03-07-15-09.json')['data'],
            '16K': load_logs(r'../logs/2022-3-1/fedstil_k_8-2022-03-06-11-00.json')['data'],
            '18K': load_logs(r'../logs/2022-3-1/fedstil_k_9-2022-03-07-21-51.json')['data'],
            '20K': load_logs(r'../logs/2022-3-1/fedstil_k_10-2022-03-06-16-58.json')['data'],
        },
        save_path_prefix=r'D:\Ryan\Projects\EdgeAI_ReID\logs\2022-3-1\abc',
        metric='val_map',
        metric_desc='mAP-Forgetting',
    )
