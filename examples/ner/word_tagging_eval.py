from os.path import expanduser, join

import numpy as np
import joblib
import torch
import torchtext.data as ttdata
from matplotlib import patches, rcParams, gridspec
from matplotlib.font_manager import FontProperties
from torchtext.data import Iterator

from exps.word_tagging import make_data, zero_num
from sdtw_pp.externals.torchtext.data import SequenceTaggingDataset, \
    NestedField
from sdtw_pp.utils import get_device
from sdtw_pp.viterbi.evaluation import iob1_iobes

import matplotlib.pyplot as plt

exp_dir = expanduser('~/output/sdtw/word_tagging')

rcParams['text.usetex'] = True
# rcParams['font.sans-serif'] = ['Verdana']


def main(run, language, pad_edge=False, cuda=False, n_samples=2048, name=''):
    cuda = cuda and torch.cuda.is_available()

    if cuda:
        device = get_device()
        print('Working on GPU %i' % device)
    else:
        device = -1
        print('Working on CPU')

    if pad_edge:
        init_token = '<init>'
        eos_token = '<end>'
    else:
        init_token = None
        eos_token = None

    tags_field = ttdata.Field(sequential=True, include_lengths=True,
                              preprocessing=iob1_iobes,
                              init_token=init_token,
                              eos_token=eos_token,
                              pad_token=None,
                              unk_token=None,
                              batch_first=True)
    sentences_field = ttdata.Field(sequential=True, include_lengths=False,
                                   batch_first=True,
                                   init_token=init_token,
                                   eos_token=eos_token,
                                   preprocessing=zero_num)
    raw_sentences_field = ttdata.Field(sequential=True, include_lengths=False,
                                       batch_first=True,
                                       init_token=init_token,
                                       eos_token=eos_token,
                                       use_vocab=True,
                                       preprocessing=None)
    letter = ttdata.Field(sequential=True, tokenize=list,
                          include_lengths=True,
                          init_token=None,
                          eos_token=None,
                          preprocessing=zero_num,
                          batch_first=True)
    letters = NestedField(letter, use_vocab=True,
                          tensor_type=torch.FloatTensor,
                          init_token=init_token,
                          eos_token=eos_token,
                          )

    model = torch.load(open(join(exp_dir, run, 'model_10.pt'), 'rb'),
                       map_location=lambda storage, loc: storage)

    if language == 'en':
        fields = [[('sentences', sentences_field),
                   ('raw_sentences', raw_sentences_field),
                   ('letters', letters)],
                  ('', None), ('', None), ('tags', tags_field)]
    elif language == 'de':
        fields = [[('sentences', sentences_field), ('letters', letters)],
                  ('', None), ('', None), ('', None), ('tags', tags_field)]
    elif language in ['es', 'nl']:
        fields = [[('sentences', sentences_field), ('letters', letters)],
                  ('', None), ('tags', tags_field)]
    else:
        raise ValueError('Wrong language')

    tagger_languages = {'en': 'eng',
                        'nl': 'ned',
                        'de': 'deu',
                        'es': 'esp'}

    test_data = SequenceTaggingDataset(
        path=expanduser('~/data/sdtw_data/conll/%s.testb'
                        % tagger_languages[language]),
        fields=fields)

    vocab = torch.load(open(join(exp_dir, run, 'vocab.pt'), 'rb'))
    sentences_field.vocab = vocab['sentences']
    letters.vocab = vocab['letters']
    letter.vocab = vocab['letters']
    tags_field.vocab = vocab['tags']
    raw_sentences_field.build_vocab(test_data)

    test_iter = Iterator(test_data,
                         shuffle=False,
                         sort_within_batch=True,
                         batch_size=n_samples,
                         repeat=False,
                         device=device)

    # old_itos = {v: k for k, v in tags_field.vocab.stoi.items()}
    old_itos = tags_field.vocab.itos
    itos = {}
    stoi = {}
    perm = []
    i = 0
    for kind in ['LOC', 'ORG', 'PER', 'MISC']:
        for loc in ['S', 'B', 'I', 'E']:
            tag = '%s-%s' % (loc, kind)
            itos[i] = tag
            stoi[tag] = i
            perm.append(tags_field.vocab.stoi[tag])
            i += 1
    tag = 'O'
    itos[i] = tag
    stoi[tag] = i
    perm.append(tags_field.vocab.stoi[tag])

    res = []
    for batch in test_iter:
        data = make_data(batch)
        raw_sentences = batch.raw_sentences
        sentences, tags, lengths, letters, letters_lengths = data
        pred_tags = model(sentences, lengths, letters, letters_lengths,
                          sorted=True)
        tags = tags.data
        pred_tags = pred_tags.data
        for tag, pred_tag, sentence, length in zip(tags, pred_tags, raw_sentences, lengths):
            this_pred = pred_tag[:length].cpu().numpy()[:, perm]
            this_tag = convert(tag[:length].cpu().numpy(), old_itos,
                               stoi)
            this_sentence = sentence[:length].data.cpu().numpy()
            this_raw_tag = denumericalize(this_tag, itos)
            this_raw_sentence = denumericalize(this_sentence,
                                               raw_sentences_field.vocab.itos)
            res.append([this_raw_sentence, this_raw_tag, this_sentence,
                        this_tag, this_pred])
    joblib.dump(res, 'res_%s.pkl' % name)
    joblib.dump(itos, 'itos_%s.pkl' % name)


def plot(ax, res, itos, k=1):
    res = res[k]
    raw_src, raw_tag, src, tag, pred = res
    pred = pred.T
    ax.imshow(pred, cmap=plt.cm.Oranges, vmin=0, vmax=1)

    for k, attn_row in enumerate(pred):
        brk = np.diff(attn_row)
        brk = np.where(brk != 0)[0]
        brk = np.append(0, brk + 1)
        brk = np.append(brk, len(attn_row))

        right_border = True
        for s, t in zip(brk[:-1], brk[1:]):
            if attn_row[s:t].sum() == 0:
                right_border = False
                continue
            lines = [(s, k), (t, k), (t, k + 1), (s, k + 1)]
            lines = np.array(lines, dtype=np.float) - 0.5
            path = patches.Polygon(lines, facecolor='none', linewidth=1.5,
                                   alpha=1, joinstyle='round',
                                   closed=not right_border,
                                   edgecolor='#999999')
            ax.add_patch(path)
            right_border = True

    ax.hlines([3.5, 7.5, 11.5, 15.5], -.5, pred.shape[1] - .5)

    ax.scatter(np.arange(pred.shape[1]), np.array(tag), s=4, c='r')

    ax.set_yticks(range(len(itos)))
    ax.set_yticklabels([itos[i] for i in range(len(itos))], fontsize=11)
    ax.set_xticks(range(len(raw_src)))
    ax.set_xticklabels(raw_src, rotation=60, ha='right', fontsize=13)


def denumericalize(seq, itos):
    return [itos[elem] for elem in seq]


def convert(seq, orig_itos, new_stoi):
    return [new_stoi[orig_itos[elem]] for elem in seq]


def plot_two(k=1):
    res_entropy = joblib.load('res_entropy.pkl')
    itos_entropy = joblib.load('itos_entropy.pkl')

    res_l2 = joblib.load('res_l2.pkl')
    itos_l2 = joblib.load('itos_l2.pkl')

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    plot(axes[0], res_entropy, itos_entropy, k)
    axes[0].annotate('Entropy regularization',
                     xy=(.5, 1.02),
                     xycoords='axes fraction',
                     ha='center', va='bottom', fontsize=16)

    axes[0].set_xticks([])
    plot(axes[1], res_l2, itos_l2, k)
    axes[1].annotate('L2 regularization',
                     xy=(.5, 1.02),
                     xycoords='axes fraction',
                     ha='center', va='bottom', fontsize=16)
    fig.subplots_adjust(bottom=0.18, top=0.93, right=0.98,
                        left=0.1,
                        hspace=0.15, )
    plt.savefig('ner_%s.pdf' % k)
    plt.show()


def plot_four(k1=1, k2=2):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                             gridspec_kw={'width_ratios': [1.35, 1]})

    res_entropy = joblib.load('res_entropy.pkl')
    itos_entropy = joblib.load('itos_entropy.pkl')

    res_l2 = joblib.load('res_l2.pkl')
    itos_l2 = joblib.load('itos_l2.pkl')

    for i, k in enumerate([k1, k2]):

        plot(axes[0, i], res_entropy, itos_entropy, k)
        axes[0, i].annotate('Entropy regularization',
                         xy=(.5, 1.02),
                         xycoords='axes fraction',
                         ha='center', va='bottom', fontsize=16)

        axes[0, i].set_xticks([])
        plot(axes[1, i], res_l2, itos_l2, k)
        axes[1, i].annotate('L2 regularization',
                         xy=(.5, 1.02),
                         xycoords='axes fraction',
                         ha='center', va='bottom', fontsize=16)
    fig.subplots_adjust(bottom=0.15, top=0.98, right=0.99,
                        left=0.05,
                        hspace=0.1, )
    plt.savefig('ner_%s.pdf' % k)
    plt.show()


if __name__ == '__main__':
    # main('38', language='en', name='entropy')
    # main('39', language='en', name='l2')
    # main('46', language='en', name='entropy')
    # main('48', language='en', name='l2')
    # main('49', language='en', name='l2')
    # for i in range(0, 1000, 10):
    #     plot_two(i)
    # plot_two(700)
    # plot_two(329)
    # plot_two(124)
    # plot_two(300)
    # plot_two(200)
    # plot_two(230)
    # plot_two(100)
    plot_four(20, 140)
