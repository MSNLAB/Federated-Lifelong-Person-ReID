# Lifelong Person Re-identification Deployed in Real-life using Spatial-Temporal Federated Learning

Data drift is a thorny challenge when deploying person re-identification (ReID) models into real-world devices, where the person data distribution is different from that of the training environment and keeps changing. To tackle the issue, we leverage both lifelong learning and federated learning techniques to continuously update ReID models deployed on many distributed edge clients. Unlike previous efforts, our solution, FedSTIL (federated spatial-temporal incremental learning), aims to mine spatial-temporal correlations among the data processed by edge devices. Specifically, the framework first periodically extracts general representations of drifted data and updates models on local edge devices. Then, the updated knowledge will be aggregated into a centralized parameter server, where the knowledge will be selectively and attentively distilled from spatial- and temporal-dimension with carefully designed mechanisms. Next, the more informative spatial-temporal knowledge will be sent back to local edge devices as a guide to further improve data representation for better performance with a lifelong learning method. Extensive experiments on a mixture of five real-world datasets demonstrate that our method outperforms others by nearly 4% in Rank-1 accuracy, while reducing communication cost by 62%. 

> Paper is under the review, the site will be announced soon.

# Quick Start

1. Please install all dependencies from `requirements.txt` after creating your own python environment with `python >= 3.8.0`.

```shell
$ pip3 install -r requirements.txt
```

2. Download our prepared simple [person re-identification datasets](https://drive.google.com/file/d/10NDQy0IZXupqXBhKfm3j7SwF08JBrE-w/view?usp=sharing), and then unzip on the project root path.  

```shell
$ tar -zxvf preprocessed_shuffle.tar.gz
```

Alternatively, you can organize your own dataset for test, please follow the structure of our dataset or use our build tool. The dataset build tools will be published soon.

3. If you have multiple GPUs, please configure in `./configs/common.json` 

```yaml
device:
  - cuda:0
  - cuda:1
  - cuda:2
  - cuda:3
  - cuda:4
```

4. Startup the experiments in `./configs/basis_exp/experiment_fedstil.yaml`

```shell
$ python3 main.py --experiments ./configs/basis_exp/experiment_fedstil.yaml
```

We have prepared various experiment settings for our given datasets, you can find them in `./configs/`. You can also set up your own experiment by following the format of given configurations.

# Results

All clients and server will save the checkpoints of model parameters and communication content on the `./ckpts/`. The log files for each experiment is on the `./logs/`, which include experimental settings, and training & evaluation performances. We also provide the analyze tools on the package `./analyse/`. Those tools may help you analyze the accuracy, forgetting, and visualization.

# Contributing

Pull requests are more than welcome! If you have any questions please feel free to contact us.

E-mail:    [guanyugao@gmail.com](mailto:guanyugao@gmail.com); [gygao@njust.edu.cn](mailto:gygao@njust.edu.cn) 

# Citation

 If you use this for research, please cite. The example BibTeX entry will be given after paper review. 

# License

Copyright 2021, MSNLAB, NUST SCE

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

