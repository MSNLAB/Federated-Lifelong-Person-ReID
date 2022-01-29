nohup python3 main.py --experiments ./configs/basis_exp/experiment_sm.yaml \
                                    ./configs/basis_exp/experiment_mm.yaml \
                                    ./configs/basis_exp/experiment_ewc.yaml \
                                    ./configs/basis_exp/experiment_mas.yaml \
                                    ./configs/basis_exp/experiment_icarl.yaml \
                                    ./configs/basis_exp/experiment_fedavg.yaml \
                                    ./configs/basis_exp/experiment_fedprox.yaml \
                                    ./configs/basis_exp/experiment_fedcurv.yaml \
                                    ./configs/basis_exp/experiment_fedweit.yaml \
                                    > task.log 2>&1 &