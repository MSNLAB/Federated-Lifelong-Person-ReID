import argparse

import yaml

from experiment import ExperimentStage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiments', type=str, nargs='+', required=True, help='Experiment yaml file path')
    args = vars(parser.parse_args())

    with open('./configs/common.yaml', 'r') as f:
        common_config = yaml.load(f, Loader=yaml.Loader)
        if not isinstance(common_config['device'], list):
            common_config['device'] = [common_config['device']]

    experiment_configs = []
    for experiment_path in args['experiments']:
        with open(experiment_path, 'r') as f:
            exp_config = dict(common_config['defaults'])
            exp_config.update(yaml.load(f, Loader=yaml.Loader))
            experiment_configs.append(exp_config)

    with ExperimentStage(common_config, experiment_configs) as exp_stage:
        exp_stage.run()
