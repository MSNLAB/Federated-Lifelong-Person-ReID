import json
import os
import random
from datetime import datetime
from typing import Dict, Tuple, Union, Any

import torch

from builder import parser_server, parser_client
from tools.logger import Logger
from tools.utils import same_seeds, clear_cache


class ExperimentLog(object):

    def __init__(self, save_path: str):
        self.records = {}
        self.save_path = save_path

    def _update_iter(self, key, value):
        keys = key.split('.')
        current_record = self.records
        for idx, key in enumerate(keys):
            if idx != len(keys) - 1:
                if key not in current_record.keys():
                    current_record[key] = {}
                current_record = current_record[key]
            else:
                if key not in current_record.keys():
                    current_record[key] = value
                else:
                    if isinstance(current_record[key], list):
                        current_record[key].append(value)
                    elif isinstance(current_record[key], set):
                        current_record[key].add(value)
                    elif isinstance(current_record[key], dict):
                        current_record[key].update(value)
                    else:
                        current_record[key] = value

    def _save_logs(self):
        dirname = os.path.dirname(self.save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(self.save_path, "w") as f:
            json.dump(self.records, f, indent=2)

    def record(self, key, value):
        self._update_iter(key, value)
        self._save_logs()


class ExperimentStage(object):

    def __init__(self, common_config: Dict, exp_configs: Union[Dict, Tuple[Dict]]):
        self.common_config = common_config
        self.exp_configs = [exp_configs] if isinstance(exp_configs, Dict) else exp_configs
        self.logger = Logger('env')

    def __enter__(self):
        clear_cache()
        same_seeds(self.common_config['random_seed'])
        self.check_environment()
        return self

    def __exit__(self, type, value, trace):
        clear_cache()
        return self

    def check_environment(self):
        # check runtime device
        device = self.common_config['device']
        try:
            torch.Tensor([0]).to(device)
        except:
            self.logger.error(f'Not available for given device {device}.')
            exit(1)

        # check dataset base path
        datasets_base = self.common_config['datasets_base']
        if not os.path.exists(datasets_base):
            self.logger.error(f'Datasets base path could not be found with {datasets_base}.')
            exit(1)

        # check dataset base path
        checkpoints = self.common_config['checkpoints']
        if os.path.exists(checkpoints):
            self.logger.warn(f'Checkpoint directory {checkpoints} is not empty.')

        self.logger.info('Experiment stage build success.')

    def run(self):
        for exp_id, exp_config in enumerate(self.exp_configs, 0):
            # generate log with time-based savepath
            format_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
            log = ExperimentLog(os.path.join(
                self.common_config['log_path'],
                f"{exp_config['name']}-{format_time}.log"
            ))
            log.record('config', exp_config)

            self.logger.info(f"Experiment loading succeed: {exp_config['name']}")
            self.logger.info(f"For more details: {log.save_path}")

            # generate server and clients
            server = parser_server(
                job_name=exp_config['name'],
                method_name=exp_config['method'],
                server_config=exp_config['server'],
                common_config=self.common_config
            )

            clients = [
                parser_client(
                    job_name=exp_config['name'],
                    method_name=exp_config['method'],
                    client_config=client_config,
                    common_config=self.common_config,
                ) for client_config in exp_config['clients']
            ]

            # simulate communication process
            for curr_round in range(1, int(exp_config['comm_rounds']) + 1):
                self._process_one_round(curr_round, server, clients, exp_config, log)

            del server, clients
            clear_cache()

    def _process_one_round(self, curr_round, server, clients, job_config, log) -> Any:
        # sample online clients
        online_clients = random.sample(clients, job_config['comm_online_clients'])
        val_intervals = job_config['val_intervals']

        # update clients with server state
        for client in online_clients:
            if client.client_name not in server.clients.keys():
                dispatch_state = server.get_dispatch_integrated_state()
                if dispatch_state is not None:
                    client.update_by_integrated_state(dispatch_state)
            else:
                dispatch_state = server.get_dispatch_incremental_state()
                if dispatch_state is not None:
                    client.update_by_incremental_state(dispatch_state)
            server.save_state(
                f'{curr_round}-{server.server_name}-{client.client_name}',
                dispatch_state, True
            )

        # simulate training for each online client
        for client in online_clients:
            task_pipeline = client.args['task_pipeline']
            task = task_pipeline.next_task()
            if task['epochs'] != 0:
                tr_output = client.train(
                    task['epochs'],
                    task['task_name'],
                    task['tr_loader'],
                    task['query_loader']
                )

                log.record(f"data.{client.client_name}.{curr_round}.{task['task_name']}", {
                    "tr_acc": tr_output['accuracy'],
                    "tr_loss": tr_output['loss']
                })

                if self.common_config['debug']:
                    cmc, mAP, avg_rep = client.validate(
                        task['task_name'],
                        task['query_loader'],
                        task['gallery_loaders']
                    )
                    log.record(f"data.{client.client_name}.{curr_round}.{task['task_name']}", {
                        "val_rank_1": cmc[0],
                        "val_rank_3": cmc[2],
                        "val_rank_5": cmc[4],
                        "val_rank_10": cmc[9],
                        "val_map": mAP,
                        'val_avg_representation': avg_rep.tolist(),
                    })

            clear_cache()

        # simulate validation for each client
        if curr_round % val_intervals == 0:
            for client in clients:
                task_pipeline = client.args['task_pipeline']
                for tid in range(len(task_pipeline.task_list)):
                    task = task_pipeline.get_task(tid)
                    cmc, mAP, avg_rep = client.validate(
                        task['task_name'],
                        task['query_loader'],
                        task['gallery_loaders']
                    )
                    log.record(f"data.{client.client_name}.{curr_round}.{task['task_name']}", {
                        "val_rank_1": cmc[0],
                        "val_rank_3": cmc[2],
                        "val_rank_5": cmc[4],
                        "val_rank_10": cmc[9],
                        "val_map": mAP,
                        'val_avg_representation': avg_rep.tolist(),
                    })

                clear_cache()

        # Communication with server
        for client in online_clients:
            incremental_state = client.get_incremental_state()
            client.save_state(
                f'{curr_round}-{client.client_name}-{server.server_name}',
                incremental_state, True
            )
            if incremental_state is not None:
                server.set_client_incremental_state(client.client_name, incremental_state)

        server.calculate()
