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
    # forgetting_on_round(
    #     logs=load_logs(r'RESULT_PATH'),
    #     rounds=60,
    #     metric='val_map',  # val_rank_1, val_rank_3, val_rank_5, val_rank_10, val_map
    #     metric_desc='mAP',  # Rank-1, Rank-3, Rank-5, Rank-10, mAP
    # )
    #
    # plot_merged_forgetting_for_many_jobs(
    #     jobs={
    #         '0K': load_logs(r'RESULT_PATH')['data'],
    #         '2K': load_logs(r'RESULT_PATH')['data'],
    #         '4K': load_logs(r'RESULT_PATH')['data'],
    #         '6K': load_logs(r'RESULT_PATH')['data'],
    #         '8K': load_logs(r'RESULT_PATH')['data'],
    #         '10K': load_logs(r'RESULT_PATH')['data'],
    #         '12K': load_logs(r'RESULT_PATH')['data'],
    #         '14K': load_logs(r'RESULT_PATH')['data'],
    #         '16K': load_logs(r'RESULT_PATH')['data'],
    #         '18K': load_logs(r'RESULT_PATH')['data'],
    #         '20K': load_logs(r'RESULT_PATH')['data'],
    #     },
    #     save_path_prefix=r'SAVE_PREFIX',
    #     metric='val_map',
    #     metric_desc='mAP-Forgetting',
    # )

    pass
