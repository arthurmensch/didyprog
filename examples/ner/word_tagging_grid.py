import os
import subprocess
from os.path import expanduser, join

import yaml
from sklearn.model_selection import ParameterGrid


def unroll_grid(grid):
    if 'grid' in grid:
        subgrids = grid.pop('grid')
        if not isinstance(subgrids, list):
            subgrids = [subgrids]
        subgrids = map(unroll_grid, subgrids)
        unrolled = []
        for subgrid in subgrids:
            for subsubgrid in subgrid:
                unrolled.append(dict(**grid, **subsubgrid))
        return unrolled
    else:
        return [grid]


config = list(yaml.load_all(open('word_tagging.yaml', 'r')))
default = config[0]['default']
grids = config[1]
exp_dir = default['system']['exp_dir']
grid_dir = expanduser(join(exp_dir, 'grid'))
if not os.path.exists(grid_dir):
    os.makedirs(grid_dir)
    off = 0
else:
    off = len(os.listdir(grid_dir))

grids = unroll_grid(grids)

config_files = []
for grid in grids:
    for key, value in grid.items():
        if not isinstance(value, (list, tuple)):
            grid[key] = [value]
    grid = ParameterGrid(grid)
    for i, param in enumerate(grid):
        config_file = join(grid_dir, 'word_tagging_%i.yaml' % (i + off))
        print(param)
        param['system'] = {'restart_config_file': config_file}
        yaml.dump(param, open(config_file, 'w+'))
        config_files.append(config_file)
    off += len(grid)

for config_file in config_files:
    subprocess.call("oarsub -S './word_tagging.sh with %s'" % config_file,
                    shell=True)
