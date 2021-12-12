import argparse
import json

from experiment import ExperimentStage

with open('./configs/common.json', 'r') as f:
    common_config = json.load(fp=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiments', type=str, nargs='+', required=True, help='Experiment json file path')
    args = vars(parser.parse_args())

    experiment_configs = []
    for experiment_path in args['experiments']:
        with open(experiment_path, 'r') as f:
            experiment_configs.append(json.load(fp=f))

    with ExperimentStage(common_config, experiment_configs) as exp_stage:
        exp_stage.run()
