import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Dict, Tuple, Union, Any

import torch

from builder import parser_server, parser_clients
from tools.logger import Logger
from tools.utils import clear_cache, same_seeds


class ExperimentLog(object):

    def __init__(self, save_path: str):
        self.records = {}
        self.save_path = save_path
        self.lock = Lock()

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
        self.lock.acquire()
        self._update_iter(key, value)
        self._save_logs()
        self.lock.release()


class VirtualContainer(object):

    def __init__(self, devices: list, parallel: int = 1) -> None:
        super().__init__()
        self.lock = Lock()
        self.devices = {device: parallel for device in devices}

    def max_worker(self):
        return sum(self.devices.values())

    def acquire_device(self):
        device = None
        self.lock.acquire()
        for dev, cnt in self.devices.items():
            if cnt and device is None:
                self.devices[dev] -= 1
                device = dev
        self.lock.release()
        return device

    def release_device(self, device):
        self.lock.acquire()
        self.devices[device] += 1
        self.lock.release()

    def possess_device(self):
        class VirtualProcess(object):

            def __init__(self, container) -> None:
                super().__init__()
                self.container = container
                self.device = None

            def __enter__(self):
                self.device = self.container.acquire_device()
                return self.device

            def __exit__(self, type, value, trace):
                self.container.release_device(self.device)
                return

        return VirtualProcess(self)


class ExperimentStage(object):

    def __init__(self, common_config: Dict, exp_configs: Union[Dict, Tuple[Dict]]):
        self.common_config = common_config
        self.exp_configs = [exp_configs] if isinstance(exp_configs, Dict) else exp_configs
        self.logger = Logger('stage')
        self.container = VirtualContainer(self.common_config['device'], self.common_config['parallel'])

    def __enter__(self):
        self.check_environment()
        return self

    def __exit__(self, type, value, trace):
        if type is not None and issubclass(type, Exception):
            self.logger.error(value)
            raise trace
        return self

    def check_environment(self):
        # check runtime device
        devices = self.common_config['device']
        for device in devices:
            try:
                torch.Tensor([0]).to(device)
            except Exception as ex:
                self.logger.error(f'Not available for given device {device}:{ex}')
                exit(1)

        # check dataset base path
        datasets_dir = self.common_config['datasets_dir']
        if not os.path.exists(datasets_dir):
            self.logger.error(f'Datasets base directory could not be found with {datasets_dir}.')
            exit(1)

        # check dataset base path
        checkpoints_dir = self.common_config['checkpoints_dir']
        if os.path.exists(checkpoints_dir):
            self.logger.warn(f'Checkpoint directory {checkpoints_dir} is not empty.')

        self.logger.info('Experiment stage build success.')

    def run(self):
        for exp_config in self.exp_configs:
            same_seeds(exp_config['random_seed'])

            # generate log with time-based savepath
            format_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
            log = ExperimentLog(os.path.join(
                self.common_config['logs_dir'],
                f"{exp_config['exp_name']}-{format_time}.json"
            ))
            log.record('config', exp_config)

            self.logger.info(f"Experiment loading succeed: {exp_config['exp_name']}")
            self.logger.info(f"For more details: {log.save_path}")

            # generate server and clients
            server = parser_server(exp_config, self.common_config)
            clients = parser_clients(exp_config, self.common_config)

            # simulate communication process
            comm_rounds = int(exp_config['exp_opts']['comm_rounds'])
            for curr_round in range(1, comm_rounds + 1):
                self.logger.info(f'Start communication round: {curr_round:0>3d}/{comm_rounds:0>3d}')
                self._process_one_round(curr_round, server, clients, exp_config, log)

            del server, clients, log
            clear_cache()

    def _process_one_round(self, curr_round, server, clients, exp_config, log) -> Any:
        # sample online clients
        online_clients = random.sample(clients, exp_config['exp_opts']['online_clients'])
        val_intervals = exp_config['exp_opts']['val_interval']

        # update clients with server state
        for client in online_clients:
            if client.client_name not in server.clients.keys():
                server.register_client(client.client_name)
                dispatch_state = server.get_dispatch_integrated_state(client.client_name)
                if dispatch_state is not None:
                    client.update_by_integrated_state(dispatch_state)
            else:
                dispatch_state = server.get_dispatch_incremental_state(client.client_name)
                if dispatch_state is not None:
                    client.update_by_incremental_state(dispatch_state)
            server.save_state(
                f'{curr_round}-{server.server_name}-{client.client_name}',
                dispatch_state, True
            )

        # simulate training for each online client
        with ThreadPoolExecutor(self.container.max_worker()) as pool:
            futures = []
            for client in online_clients:
                futures.append(pool.submit(
                    self._process_train,
                    *(client, log, curr_round, self.container)
                ))
            for future in as_completed(futures):
                future.result(timeout=1800)
                if future.exception():
                    raise future.exception()

        # simulate validation for each client
        if curr_round % val_intervals == 0:
            with ThreadPoolExecutor(self.container.max_worker()) as pool:
                futures = []
                for client in clients:
                    futures.append(pool.submit(
                        self._process_val,
                        *(client, log, curr_round, self.container)
                    ))
                for future in as_completed(futures):
                    future.result(timeout=1800)
                    if future.exception():
                        raise future.exception()

        # communication with server
        for client in online_clients:
            incremental_state = client.get_incremental_state()
            client.save_state(
                f'{curr_round}-{client.client_name}-{server.server_name}',
                incremental_state, True
            )
            if incremental_state is not None:
                server.set_client_incremental_state(client.client_name, incremental_state)

        server.calculate()

    @staticmethod
    def _process_train(client, log, curr_round, container):
        with container.possess_device() as device:
            try:
                task_pipeline = client.task_pipeline
                task = task_pipeline.next_task()
                if task['tr_epochs'] != 0:
                    tr_output = client.train(
                        epochs=task['tr_epochs'],
                        task_name=task['task_name'],
                        tr_loader=task['tr_loader'],
                        val_loader=task['query_loader'],
                        device=device
                    )
                    log.record(f"data.{client.client_name}.{curr_round}.{task['task_name']}", {
                        "tr_acc": tr_output['accuracy'],
                        "tr_loss": tr_output['loss']
                    })
            except Exception as ex:
                client.logger.error(ex)
                raise ex
            finally:
                clear_cache()

    @staticmethod
    def _process_val(client, log, curr_round, container):
        with container.possess_device() as device:
            try:
                task_pipeline = client.task_pipeline
                for tid in range(len(task_pipeline.task_list)):
                    task = task_pipeline.get_task(tid)
                    cmc, mAP, avg_rep = client.validate(
                        task_name=task['task_name'],
                        query_loader=task['query_loader'],
                        gallery_loader=task['gallery_loaders'],
                        device=device
                    )
                    log.record(f"data.{client.client_name}.{curr_round}.{task['task_name']}", {
                        "val_rank_1": cmc[0],
                        "val_rank_3": cmc[2],
                        "val_rank_5": cmc[4],
                        "val_rank_10": cmc[9],
                        "val_map": mAP,
                        'val_avg_representation': avg_rep.tolist(),
                    })
            except Exception as ex:
                client.logger.error(ex)
                raise ex
            finally:
                clear_cache()
