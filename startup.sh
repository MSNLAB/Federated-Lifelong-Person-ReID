nohup python3 main.py --experiments ./configs/experiment_sm.yaml \
                                    ./configs/experiment_mm.yaml \
                                    ./configs/experiment_ewc.yaml \
                                    ./configs/experiment_mas.yaml \
                                    ./configs/experiment_fedavg.yaml \
                                    ./configs/experiment_fedprox.yaml \
                                    ./configs/experiment_fedcurv.yaml \
                                    ./configs/experiment_fedweit.yaml \
                                    > task.log 2>&1 &