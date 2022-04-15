import json

color = ['indianred', 'mediumseagreen', 'darkorchid', 'burlywood', 'orange', 'cornflowerblue']
line_style = ['-', '--', ':', '-.', '--', ':', '-.', '--', ':', '-.', ]
marker = ['s', 'o', '^', 'P', '*', 'D', '|', 'v', 'x', '8']


def load_logs(log_path: str):
    with open(log_path, 'r') as f:
        state = json.load(fp=f)
    return state
