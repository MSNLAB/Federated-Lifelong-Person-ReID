This directory contains the checkpoints of all experiments.

Please make sure that the checkpoint directories are clear before all experiments, otherwise the old checkpoint buffers would affect the new experiments.

The checkpoints consist with:
> 1. Model Parameters;
> 2. Communication Content of S2C or C2S.

The structure of checkpoint directory is:
```shell
./ckpts/{Experiment_Name}/{Client_or_Server_Name}/{Checkpoint_Name}.ckpt
```

