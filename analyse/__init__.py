import json

color = ['indianred', 'mediumseagreen', 'darkorchid', 'burlywood', 'orange', 'cornflowerblue']
line_style = ['-', '--', ':', '-.', '--', ':', '-.', '--', ':', '-.', ]
marker = ['s', 'o', '^', 'P', '*', 'D', '|', 'v', 'x', '8']

metrics = ['val_rank_1', 'val_rank_3', 'val_rank_5', 'val_map']
metrics_desc = ['Rank-1', 'Rank-3', 'Rank-5', 'mAP']


def load_logs(log_path: str):
    with open(log_path, 'r') as f:
        state = json.load(fp=f)
    return state
