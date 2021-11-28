import datetime
import json
import os
import random
from typing import Dict, Tuple, Union, Any

import torch

from builder import parser_server, parser_client
from tools.utils import same_seeds


class ExperimentStage(object):

    def __init__(self, common_config: Dict, job_configs: Union[Dict, Tuple[Dict]]):
        self.common_config = common_config
        self.job_configs = [job_configs] if isinstance(job_configs, Dict) else job_configs
        self.record = {}

    def __enter__(self):
        self._empty_cuda_cache()
        same_seeds(self.common_config['random_seed'])
        return self

    def __exit__(self, type, value, trace):
        self._empty_cuda_cache()
        return self

    def _empty_cuda_cache(self):
        if 'cuda' in self.common_config['device']:
            torch.cuda.empty_cache()

    def performance(self):
        for job_id, job_config in enumerate(self.job_configs, 0):
            self._empty_cuda_cache()
            self.record = {}

            # Generate server and clients
            server = parser_server(
                job_name=job_config['name'],
                method_name=job_config['method'],
                server_config=job_config['server'],
                common_config=self.common_config
            )
            clients = [
                parser_client(
                    job_name=job_config['name'],
                    method_name=job_config['method'],
                    client_config=client_config,
                    common_config=self.common_config,
                ) for client_config in job_config['clients']
            ]

            # Register clients in server
            for client in clients:
                server.register_client(client.client_name)
                dispatch_state = server.get_dispatch_integrated_state()
                server.save_state(f's2c-0', dispatch_state, True)
                if dispatch_state is not None:
                    client.update_by_integrated_state(dispatch_state)
                self.record[f"{client.client_name}"] = {}

            # Simulate communication process
            for comm in range(1, int(job_config['comm_rounds']) + 1):
                self._performance_one_comm(comm, server, clients, job_config)

            # Save the record
            format_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            log_save_path = os.path.join(
                self.common_config['log_path'],
                f"{job_config['name']}_{format_time}.log"
            )
            if not os.path.exists(self.common_config['log_path']):
                os.makedirs(self.common_config['log_path'])
            with open(log_save_path, "w") as f:
                json.dump(self.record, f, indent=2)

    def _performance_one_comm(self, comm_round: int, server, clients, job_config: Dict) -> Any:
        # Initial record for each client at current communication round
        for client in clients:
            self.record[f"{client.client_name}"][f"{comm_round}"] = {}

        # Sample clients as online for performance
        online_clients = random.sample(clients, job_config['comm_online_clients'])
        val_intervals = job_config['val_intervals']

        # Update clients with server state
        for client in online_clients:
            dispatch_state = server.get_dispatch_incremental_state()
            server.save_state(f's2c-{comm_round}', dispatch_state, True)
            if dispatch_state is not None:
                client.update_by_incremental_state(dispatch_state)

        # Simulate training for each online client
        for client in online_clients:
            self._empty_cuda_cache()
            task_pipeline = client.args['task_pipeline']
            task = task_pipeline.next_task()
            if task['epochs'] != 0:
                tr_output = client.train(
                    task['epochs'],
                    task['task_name'],
                    task['tr_loader'],
                    task['query_loader']
                )
                self.record[f"{client.client_name}"][f"{comm_round}"][f"{task['task_name']}"] = {
                    "tr_acc": tr_output['accuracy'],
                    "tr_loss": tr_output['loss'],
                }

        # Simulate validation for each client
        if comm_round % val_intervals == 0:
            for client in clients:
                self._empty_cuda_cache()
                task_pipeline = client.args['task_pipeline']

                # validate all tasks
                for tid in range(len(task_pipeline.task_list)):
                    task = task_pipeline.get_task(tid)
                    cmc, mAP, avg_representation = client.validate(
                        task['task_name'],
                        task['query_loader'],
                        task['gallery_loaders']
                    )
                    self.record[f"{client.client_name}"][f"{comm_round}"][f"{task['task_name']}"] = {
                        "val_rank_1": cmc[0],
                        "val_rank_3": cmc[2],
                        "val_rank_5": cmc[4],
                        "val_rank_10": cmc[9],
                        "val_map": mAP,
                        'val_avg_representation': avg_representation.tolist(),
                    }

        # Communication with server
        for client in online_clients:
            incremental_state = client.get_incremental_state()
            client.save_state(f'c2s-{comm_round}', incremental_state, True)
            if incremental_state is not None:
                server.set_client_incremental_state(client.client_name, incremental_state)

        server.calculate()
