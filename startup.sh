nohup python3 main.py --experiments ./configs/af/experiment_sm.yaml \
                                    ./configs/af/experiment_mm.yaml \
                                    ./configs/af/experiment_ewc.yaml \
                                    ./configs/af/experiment_mas.yaml \
                                    ./configs/af/experiment_fedavg.yaml \
                                    ./configs/af/experiment_fedprox.yaml \
                                    ./configs/af/experiment_fedcurv.yaml \
                                    ./configs/af/experiment_fedweit.yaml \
                                    ./configs/af/experiment_fedstil.yaml \
                                    > task.log 2>&1 &