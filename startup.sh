export PYTHONPATH=/home/zhanglei/projects/EdgeAI_ReID

export CUDA_VISIBLE_DEVICES=0

nohup python3 main.py --experiments ./configs/experiment_sm.json \
                                    ./configs/experiment_mm.json \
                                    ./configs/experiment_fedavg.json\
                                    ./configs/experiment_fedprox.json \
                                    ./configs/experiment_ewc.json \
                                    ./configs/experiment_mas.json \
                                    > task-1.log 2>&1 &

nohup python3 main.py --experiments ./configs/experiment_fedweit.json \
                                    ./configs/experiment_fedcurv.json \
                                    > task-2.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1

nohup python3 main.py --experiments ./configs/experiment_ours_mm.json \
                                    > task-ours-mm.log 2>&1 &

nohup python3 main.py --experiments ./configs/experiment_ours_sm.json \
                                    > task-ours-sm.log 2>&1 &