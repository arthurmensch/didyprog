import codecs
import os
import shutil
from os.path import expanduser, join
from tempfile import mkdtemp

import numpy as np

eval_script = expanduser('~/work/repos/soft-dtw-pp/scripts/conlleval')


def ner_score(tags, pred_tags, tag_itos, format='iob'):
    predictions = []
    for these_tags, these_pred_tags in zip(tags, pred_tags):
        these_pred_tags = np.argmax(these_pred_tags, axis=1)
        these_tags = [tag_itos[tag] for tag in these_tags]
        these_pred_tags = [tag_itos[tag] for tag in these_pred_tags]
        if format == 'iobes':
            these_tags = iobes_iob2(these_tags)
            these_pred_tags = iobes_iob2(these_pred_tags)
        for tag, pred_tag in zip(these_tags, these_pred_tags):
            predictions.append("W %s %s" % (tag, pred_tag))
        predictions.append("")

    temp_dir = mkdtemp()
    temp_out = join(temp_dir, 'out')
    temp_in = join(temp_dir, 'in')
    with open(temp_in, 'w+') as f:
        f.write("\n".join(predictions))
    os.system("%s < %s > %s" % (eval_script, temp_in, temp_out))
    eval_lines = [l.rstrip() for l in codecs.open(temp_out, 'r', 'utf8')]
    shutil.rmtree(temp_dir)
    if len(eval_lines) > 2:
        scores = eval_lines[1].strip().split()
        fscore = float(scores[-1]) / 100
        prec = float(scores[-5][:-2]) / 100
        recall = float(scores[-3][:-2]) / 100
    else:
        prec, recall, fscore = -1, -1, -1
    return prec, recall, fscore


def iob1_iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        else:
            split = tag.split('-')
            if len(split) != 2 or split[0] not in ['I', 'B']:
                raise Exception('Invalid IOB format!')
            elif split[0] == 'B':
                new_tags.append(tag)
            elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
                new_tags.append('B' + tag[1:])
            elif tags[i - 1][1:] == tag[1:]:
                new_tags.append(tag)
            else:  # conversion IOB1 to IOB2
                new_tags.append('B' + tag[1:])
    return new_tags


def iob1_iobes(tags):
    return iob2_iobes(iob1_iob2(tags))


def iob2_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob2(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags