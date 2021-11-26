export PYTHONPATH=/home/zhanglei/projects/EdgeAI_ReID

export CUDA_VISIBLE_DEVICES=0

nohup python3 main.py --experiments ./configs/experiment_sm.json \
                                    ./configs/experiment_mm.json \
                                    ./configs/experiment_fedavg.json\
                                    > task-1.log 2>&1 &

nohup python3 main.py --experiments ./configs/experiment_ewc.json \
                                    ./configs/experiment_mas.json \
                                    ./configs/experiment_fedprox.json \
                                    > task-2.log 2>&1 &

nohup python3 main.py --experiments ./configs/experiment_fedweit.json \
                                    > task-3.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1

nohup python3 main.py --experiments ./configs/experiment_fedcurv.json \
                                    > task-4.log 2>&1 &

nohup python3 main.py --experiments ./configs/experiment_ours_kd_0.json \
                                    > task-ours-1.log 2>&1 &

nohup python3 main.py --experiments ./configs/experiment_ours_kd_1e3.json \
                                    > task-ours-2.log 2>&1 &