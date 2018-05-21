import pandas as pd

import json
import re

import os
from os.path import join, expanduser

import yaml

config = next(yaml.load_all(open('word_tagging.yaml', 'r')))
exp_dirs = [expanduser(config['default']['system']['exp_dir'])]
# exp_dirs = [expanduser('~/output/sdtw/old/word_tagging_recode'),
#            expanduser('~/output/sdtw/word_tagging_eng')
#           ]

regex = re.compile(r'[0-9]+$')
res = []
for exp_dir in exp_dirs:
    for this_dir in filter(regex.match, os.listdir(exp_dir)):
        this_exp_dir = join(exp_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
            info = json.load(open(join(this_exp_dir, 'info.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        try:
            test_score = max(info['test_f1'])
            val_score = max(info['val_f1'])
            train_score = info['train_test_f1'][-1]
            epoch = info['epochs'][-1]
        except (IndexError, KeyError, ValueError):
            test_score, val_score, train_score, epoch = 0, 0, 0, 0
        status = run['status']

        res.append(dict(test_score=test_score, val_score=val_score,
                        train_score=train_score, id=this_dir,
                        epoch=epoch, status=status, info=info, **config))
df = pd.DataFrame(res)
pd.to_pickle(df, join(exp_dirs[0], 'results.pt'))
