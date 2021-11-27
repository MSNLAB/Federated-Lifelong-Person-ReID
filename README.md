# Federated Continual Learning in Person Re-identification

## Abstract

> Supplementing...

## Quick Start

1. Install dependencies from `requirements.txt` after creating your own environment with `python >= 3.8.0`

```shell
$ pip3 install -r requirements.txt
```

2. Prepare person re-identification datasets for federated continual learning as [Awesome-ReID-for-FCL](https://github.com/MSNLAB/Awesome-ReID-for-FCL)

```shell
$ git clone https://github.com/MSNLAB/Awesome-ReID-for-FCL.git
$ python3 Awesome-ReID-for-FCL/main.py \
        --datasets market1501 duke prid2011 pku \
        --roots ./datasets/Market-1501 \
                ./datasets/DukeMTMC-reID \
                ./datasets/prid_2011 \
                ./datasets/pku_reid \
        --output ./datasets/preprocessed \
        --split_indice 0.8 0.1 0.7 \
        --task_indice 5 8 \
        --random_seed 123
```

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

4. Startup the default experiments in `./configs/experiments_XXX.json`

```shell
$ nohup python3 main.py --experiments \
				./configs/experiment_ours_sm.json \
				./configs/experiment_ours_mm.json \
				> task.log 2>&1 &
$ tail -f task.log
```

## Manual Experiment

You can easily modify the hyper-parameters as you need and implement a new method. Before this, we would like to introduce the general structure of this project and help you revise the experiment.

```
|-- configs
    |-- common.json           `enviroment configurations such as device`
    |-- experiment_xxx.json   `experiment configurations such as hyper-params`	
|-- criterions
    |-- xxx.py                `loss functions`
|-- datasets
    |-- datasets_loader.py    `the dataset class for reading sources`
    |-- datasets_pipeline.py  `the pipeline class for simulating tasks stream`
    |-- image_augmentation.py `image augmentation methods`
|-- methods
    |-- xxx.py                `the most important algorithm structure`
|-- models
    |-- resnet.py             `resnet that meet re-identification requirments`
|-- modules
    |-- client.py             `define the client functions that methods must be done`
    |-- server.py             `define the server functions that methods must be done`
    |-- operator.py           `define the other functions that methods must be done`
|-- tools
    |-- distance.py           `the method to calculate similarity of representations`
    |-- evaluate.py           `the method to calculate rank-K and mAP`
    |-- logger.py             `the method to print logs`
    |-- winit.py              `the method to init the model parameters`
    |-- utils.py              `the other utils functions for recalling`
|-- main.py                   `reading configs from command parameters`
|-- builder.py                `construct the classes by configuration`
|-- experiment.py             `experiment stage for all methods`
|-- analysis.py               `analysis the experiment results`
```

1. Design manual experiments

   > Supplementing...

2. Design manual methods

   > Supplementing...
