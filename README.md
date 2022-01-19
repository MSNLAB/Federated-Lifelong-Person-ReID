# Federated Continual Learning in Person Re-identification

> Supplementing...

# Quick Start

1. Install dependencies from `requirements.txt` after creating your own environment with `python >= 3.8.0`

```shell
$ pip3 install -r requirements.txt
```

2. Prepare person re-identification datasets for federated continual learning as [Awesome-ReID-for-FCL](https://github.com/MSNLAB/Awesome-ReID-for-FCL).  

```shell
$ git clone https://github.com/MSNLAB/Awesome-ReID-for-FCL.git
$ python3 Awesome-ReID-for-FCL/main.py \
    --datasets market1501 duke prid2011 pku cuhk03 ethz \
    --roots ./datasets/Market-1501 \
            ./datasets/DukeMTMC-reID \
            ./datasets/prid_2011 \
            ./datasets/pku_reid \
            ./datasets/CUHK-03 \
            ./datasets/ethz \
    --output ./datasets/preprocessed \
    --split_indice 0.8 0.1 0.7 \
    --task_indice 5 10 \
    --temporal_indice 0.5 3.0 \
    --random_seed 123
```

*[Note] Alternatively, you can download our preprocessed open dataset for simple test, the site will be announced soon.* 

3. Configure environment in `./configs/common.json` 

```json
{
  "device": "cuda",
  "datasets_base": "/your/path/of/datasets",
  "checkpoints": "/your/path/of/checkpoints",
  "log_path": "/your/path/of/logs",
  "random_seed": 123
}
```

4. Startup the default experiments in `./configs/experiments_###.json`

```shell
$ python3 main.py --experiments \
                  ./configs/experiment_ours_sm.json \
                  ./configs/experiment_ours_mm.json
```

# Manual Experiment

You can easily modify the hyper-parameters as you need and implement a new method. Before this, we would like to introduce the general structure of this project and help you revise the experiment.

```
|-- configs
    |-- common.json               `enviroment configurations such as device`
    |-- experiment_###.json       `experiment configurations such as hyper-params`	
|-- criterions
    |-- xxx.py                    `loss functions`
|-- datasets
    |-- datasets_loader.py        `the dataset class for reading sources`
    |-- datasets_pipeline.py      `the pipeline class for simulating tasks stream`
    |-- image_augmentation.py     `image augmentation methods`
|-- methods
    |-- ###.py                    `the most important algorithm structure`
|-- models
    |-- resnet.py                 `resnet that meet re-identification requirments`
|-- modules
    |-- client.py                 `define the client functions that methods must be done`
    |-- server.py                 `define the server functions that methods must be done`
    |-- operator.py               `define the other functions that methods must be done`
|-- tools
    |-- distance.py               `the method to calculate similarity of representations`
    |-- evaluate.py               `the method to calculate rank-K and mAP`
    |-- logger.py                 `the method to print logs`
    |-- winit.py                  `the method to init the model parameters`
    |-- utils.py                  `other utils and functions for calling`
|-- main.py                       `reading configuration from command parameters`
|-- builder.py                    `construct the classes by configuration`
|-- experiment.py                 `experiment stage for all methods`
|-- analysis.py                   `analysis the experiment results`
```

1. Design manual experiments

   If you need modify or add manual experiment, please create a new json file in `./configs/experiment_{NAME}.json`.

   The format is as follows:

   ```json
   {
       "name": "your experiment name [string]",
       "method": "methods such as 'ewc', 'fedavg', 'fedcurv' [string]",
       "comm_rounds": "the communication rounds [integer]",
       "comm_online_clients": "random choose some client online each round [integer]",
       "val_intervals": "validate the Rank-K and mAP with some interval [integer]",
       "server": {
           "name": "server name [string]",
           "model": {
               "name": "model network structure such as 'resnet' [string]",
               "arguments": "model arguments [map]"
           },
           "criterion": [{
                "name": "loss function name",
                "arguments": "loss function arguments [map]"
           }],
           "optimizer": {
               "name": "optimizer name such as Adam [string]",
               "arguments": "loss function arguments [map]",
               "fine_tuning": "fine tuning flag [boolean]",
               "fine_tuning_layers": "module that need to train [list]"
           },
           "scheduler": {
               "name": "learning rate scheduler name such as step_lr",
               "arguments": "lr scheduler function arguments [map]"
           }
       },
       "clients": [
           {
               "name": "client name [string]",
               "model": "same as server format [map]",
               "criterion": "same as server format [map]",
               "optimizer": "same as server format [map]",
               "scheduler": "same as server format [map]",
               "workers": "dataloader workers number [integer]",
               "pin_memory": "dataloader pin memory [boolean]",
               "tasks": [{
                   "task_name": "your task name",
                   "epochs": "training epochs each round [integer]",
                   "batch_size": "batch size [integer]",
                   "sustain_round": "the duration of this task for epoch [integer]",
                   "img_size": [ 128, 64 ],
                   "norm_mean": [ 0.485, 0.456, 0.406 ],
                   "norm_std": [ 0.229, 0.224, 0.225 ],
                   "datasets": "dataset name , exist in common.json -> 'datasets_base' [string]",
                   "augmentation": "level of : 'none', 'defualt', 'rose', 'sharp', 'drastic' [string]"
                }]
           }
       ]
   }
   ```

2. Design manual methods

   > Supplementing...

# Contributing

Pull requests are more than welcome! If you have any questions please feel free to contact us.

E-mail:  

# License

Copyright 2021, MSNLAB, NUST SCE

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

