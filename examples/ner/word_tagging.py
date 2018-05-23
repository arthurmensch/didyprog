import time
from collections import Counter
from math import floor

import functools
import os
import random
import re
import torch
import torchtext.data as ttdata
import yaml
from didypro.ner.externals.sacred import lazy_add_artifact
from didypro.ner.externals.torchtext.data import SequenceTaggingDataset, \
    NestedField, CaseInsensitiveVectors
from didypro.ner.evaluation import ner_score, iob1_iobes
from didypro.ner.loss import BinaryMSELoss, OurNLLLoss
from didypro.ner.model import Tagger
from os.path import expanduser, join
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchtext.data import Iterator
from torchtext.vocab import Vocab, GloVe, FastText

exp_name = 'word_tagging'
exp = Experiment(name=exp_name)

config = next(yaml.load_all(open('word_tagging.yaml', 'r')))['default']
exp_dir = expanduser(config['system']['exp_dir'])
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

exp.add_config(config)
exp.observers.append(FileStorageObserver.create(basedir=expanduser(
    config['system']['exp_dir'])))


def validate(model, data_iter, score_function, objective, loss_function,
             ):
    all_tags = []
    all_pred_tags = []
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_iter:
            data = make_data(batch)
            sentences, tags, lengths, letters, letters_lengths = data
            pred_tags = model(sentences, lengths, letters, letters_lengths,
                              sorted=True)
            loss = compute_loss(model, data, objective, loss_function)
            total_loss += loss * len(sentences)
            total_samples += len(sentences)
            for tag, pred_tag, length in zip(tags, pred_tags, lengths):
                all_tags.append(tag.cpu().numpy())
                all_pred_tags.append(pred_tag.cpu().numpy())
    loss = total_loss / total_samples
    prec, recall, f1 = score_function(all_tags, all_pred_tags)
    return loss, prec, recall, f1


def compute_loss(model, data, objective, loss_function):
    sentences, tags, lengths, letters, letters_lengths = data
    if objective == 'nll':
        partition, potentials = model.partition_potentials(sentences, lengths,
                                                           letters,
                                                           letters_lengths,
                                                           sorted=True)
        n_states = potentials.shape[2]
        masks, masks_term = masks_from_tags(tags, lengths, n_states,
                                            )
        gold_score = torch.sum(torch.masked_select(potentials, masks))
        loss = torch.sum(partition) - gold_score
        loss /= sentences.shape[0]
    else:   # objective == 'erm':
        pred_tags = model(sentences, lengths, letters, letters_lengths,
                          sorted=True)
        loss = loss_function(pred_tags, tags, lengths)
    return loss


@exp.capture(prefix='system')
def init_system(exp_dir, temp_dir, cache_dir,
                cuda, _seed, _log, _run):
    exp_dir = expanduser(exp_dir)
    temp_dir = expanduser(temp_dir)
    cache_dir = expanduser(cache_dir)
    artifact_dir = join(exp_dir, str(_run._id))
    for this_dir in [exp_dir, temp_dir, cache_dir, artifact_dir]:
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
    _run.info['artifact_dir'] = artifact_dir

    torch.manual_seed(_seed)
    random.seed(_seed)

    cuda = cuda and torch.cuda.is_available()
    device = torch.device('cuda:0') if cuda else torch.device('cpu')
    _log.info('Working on device %s' % device)

    return device


@exp.capture
def dump_model(model, name, _log, _run):
    artifact_dir = _run.info['artifact_dir']
    filename = join(artifact_dir, name)
    _log.info('Dumping model at %s' % filename)
    with open(filename, 'wb+') as f:
        torch.save(model, f)
    lazy_add_artifact(_run, name, filename)
    return filename


def masks_from_tags(tags, lengths, n_states):
    n_batch, seq_len = tags.shape
    masks = torch.ByteTensor(n_batch, seq_len, n_states, n_states,
                             ).zero_()
    masks_term = torch.ByteTensor(n_batch, n_states,
                                  ).zero_()
    for b in range(n_batch):
        masks[b, 0, tags[b, 0], 0] = 1
        for t in range(1, lengths[b]):
            masks[b, t, tags[b, t], tags[b, t - 1]] = 1
        masks_term[b, tags[b, lengths[b] - 1]] = 1
    masks = masks.to(tags.device)
    masks_term = masks_term.to(tags.device)
    return masks, masks_term


def make_data(batch, augment=False,
              singleton_idx=None, unk_idx=None,
              ):
    sentences = batch.sentences
    tags, lengths = batch.tags

    letters, letters_lengths = batch.letters
    # Data augmentation for <unk> embedding training
    if augment:
        indices = torch.zeros_like(tags)
        bernoulli = torch.FloatTensor(*tags.shape,).fill_(.3)
        bernoulli = torch.bernoulli(bernoulli).byte()
        bernoulli = bernoulli.to(tags.device)
        indices = indices.byte()
        for rep in singleton_idx:
            indices = indices | (tags == rep)
        indices = indices & bernoulli
        sentences[indices] = unk_idx

    return sentences, tags, lengths, letters, letters_lengths


def zero_num(seq):
    return [re.sub('\d', '0', x) for x in seq]


@exp.automain
def main(language, hidden_dim,
         dropout, proc, letter_proc,
         objective, operator, alpha, lr, momentum,
         optimizer, batch_size, n_epochs,
         pretrained_embeddings,
         letter_hidden_dim, letter_embedding_dim, n_samples,
         pad_edge, augment,
         _seed, _run, _log):
    if objective not in ['erm', 'nll']:
        raise ValueError("`objective` should be in ['erm', 'nll'],"
                         "got %s" % objective)

    # Technical
    device = init_system()

    if pad_edge:
        init_token = '<init>'
        eos_token = '<end>'
    else:
        init_token = None
        eos_token = None
    # Data loading using torchtext abstraction
    tags = ttdata.Field(sequential=True, include_lengths=True,
                        preprocessing=iob1_iobes,
                        init_token=init_token,
                        eos_token=eos_token,
                        pad_token=None,
                        unk_token=None,
                        batch_first=True)
    sentences = ttdata.Field(sequential=True, include_lengths=False,
                             batch_first=True,
                             init_token=init_token,
                             eos_token=eos_token,
                             preprocessing=zero_num)
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

    if language == 'en':
        fields = [[('sentences', sentences), ('letters', letters)],
                  ('', None), ('', None), ('tags', tags)]
    elif language == 'de':
        fields = [[('sentences', sentences), ('letters', letters)],
                  ('', None), ('', None), ('', None), ('tags', tags)]
    elif language in ['es', 'nl']:
        fields = [[('sentences', sentences), ('letters', letters)],
                  ('', None), ('tags', tags)]
    else:
        raise ValueError('Wrong language')

    tagger_languages = {'en': 'eng',
                        'nl': 'ned',
                        'de': 'deu',
                        'es': 'esp'}

    train_data, val_data, test_data = SequenceTaggingDataset.splits(
        path=expanduser('~/data/sdtw_data/conll'),
        train='%s.train' % tagger_languages[language],
        validation='%s.testa' % tagger_languages[language],
        test='%s.testb' % tagger_languages[language],
        n_samples=n_samples,
        fields=fields)

    letters.build_vocab(train_data, val_data, test_data)
    tags.build_vocab(train_data)
    tag_itos = tags.vocab.itos
    if pad_edge:
        eos_idx = tags.vocab.stoi[tags.eos_token]
        init_idx = tags.vocab.stoi[tags.init_token]
        tag_itos[eos_idx] = 'O'
        tag_itos[init_idx] = 'O'
    else:
        eos_idx = None
        init_idx = None

    if isinstance(pretrained_embeddings, int):
        sentences.build_vocab(train_data, val_data, test_data)
        embedding_dim = pretrained_embeddings
    else:
        if pretrained_embeddings == 'ner':
            vectors = CaseInsensitiveVectors(
                expanduser('~/data/sdtw_data/ner/%s' %
                           tagger_languages[language]),
                unk_init=lambda x: x.normal_(0, 1),
                cache=expanduser('~/cache'))
        elif 'glove' in pretrained_embeddings:
            _, name, dim = pretrained_embeddings.split('.')
            dim = dim[:-1]
            GloVe.__getitem__ = CaseInsensitiveVectors.__getitem__
            vectors = GloVe(name=name, dim=dim, cache=expanduser('~/cache'))
        elif pretrained_embeddings == 'fasttext':
            FastText.__getitem__ = CaseInsensitiveVectors.__getitem__
            FastText.cache = CaseInsensitiveVectors.cache
            vectors = FastText(language=language,
                               cache=expanduser('~/cache'))
        # extend vocab with words of test/val set that has embeddings in
        # pre-trained embedding
        # A prod-version would do it dynamically at inference time
        counter = Counter()
        sentences.build_vocab(val_data, test_data)
        for word in sentences.vocab.stoi:
            if word in vectors.stoi or word.lower() in vectors.stoi or \
                    re.sub('\d', '0', word.lower()) in vectors.stoi:
                counter[word] = 1
        eval_vocab = Vocab(counter)
        print("%i/%i eval/test word in pretrained" % (len(counter),
                                                      len(sentences.vocab.stoi)))
        sentences.build_vocab(train_data)
        prev_vocab_size = len(sentences.vocab.stoi)
        sentences.vocab.extend(eval_vocab)
        new_vocab_size = len(sentences.vocab.stoi)
        print('New vocab size: %i (was %i)' % (new_vocab_size,
                                               prev_vocab_size))
        sentences.vocab.load_vectors(vectors)
        embedding_dim = sentences.vocab.vectors.shape[1]
    artifact_dir = _run.info['artifact_dir']
    vocab_dict = {'sentences': sentences.vocab,
                  'tags': tags.vocab,
                  'letters': letter.vocab}
    torch.save(vocab_dict, open(join(artifact_dir, 'vocab.pt'), 'wb+'))

    unk_idx = sentences.vocab.stoi[sentences.unk_token]
    padding_idx = sentences.vocab.stoi[sentences.pad_token]
    singleton_idx = [tags.vocab.stoi[singleton]
                     for singleton in tags.vocab.stoi if 'S-' in singleton]
    tagset_size = len(tags.vocab)
    vocab_size = len(sentences.vocab)
    letter_size = len(letters.vocab)

    device_iter = -1 if device.type == 'cpu' else device.index
    train_iter, val_iter, test_iter = Iterator.splits(
        (train_data, val_data, test_data),
        sort_within_batch=True,
        batch_sizes=(batch_size, 512, 512),
        device=device_iter)
    train_test_iter = Iterator(train_data, sort_within_batch=True,
                               batch_size=512,
                               shuffle=True,
                               device=device_iter)
    eval_iter = {'val': val_iter,
                 'test': test_iter,
                 'train_test': [next(iter(train_test_iter))]
                 }

    model = Tagger(embedding_dim, vocab_size, tagset_size,
                   hidden_dim=hidden_dim,
                   proc=proc,
                   padding_idx=padding_idx,
                   letter_proc=letter_proc,
                   letter_embedding_dim=letter_embedding_dim,
                   letter_hidden_dim=letter_hidden_dim,
                   letter_size=letter_size,
                   dropout=dropout,
                   eos_idx=eos_idx, init_idx=init_idx,
                   alpha=alpha,
                   operator=operator)

    # Load vectors
    if hasattr(sentences.vocab, 'vectors'):
        model.embedder.word_embeddings.weight.data = sentences.vocab.vectors
        model.embedder.word_embeddings.weight.data[padding_idx].fill_(0.)

    model = model.to(device=device)

    if operator == 'softmax':
        loss_function = OurNLLLoss()
    else:
        loss_function = BinaryMSELoss()

    score_function = functools.partial(ner_score, tag_itos=tag_itos,
                                       format='iobes')

    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr * batch_size,
                                    momentum=momentum)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    else:
        raise ValueError()
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=5,
                                  threshold=1e-3, cooldown=2)

    for fold in eval_iter:
        _run.info['%s_loss' % fold] = []
        _run.info['%s_prec' % fold] = []
        _run.info['%s_recall' % fold] = []
        _run.info['%s_f1' % fold] = []
    _run.info['epochs'] = []
    _run.info['time'] = []
    last_epoch = floor(train_iter.epoch)
    t0 = time.clock()
    total_time = 0

    for batch in train_iter:
        epoch = floor(train_iter.epoch)
        if epoch > last_epoch:
            t1 = time.clock()
            elapsed = t1 - t0
            total_time += elapsed
            model.eval()
            _log.info("epoch %i, time/epoch %.3f s" % (epoch, elapsed))
            if epoch % 10 == 0:
                dump_model(model, 'model_%i.pt' % epoch)
            for fold in eval_iter:
                this_iter = eval_iter[fold]
                this_iter = iter(this_iter)
                loss, prec, recall, f1 = validate(model, this_iter,
                                                  score_function, objective,
                                                  loss_function)
                if fold == 'val':
                    scheduler.step(loss.item(), epoch=epoch)
                _log.info("%s: loss %.4f, prec %.4f, recall %.4f, f1 %.4f"
                          % (fold, loss, prec, recall, f1))
                _run.info['%s_loss' % fold].append(loss.item())
                _run.info['%s_prec' % fold].append(prec)
                _run.info['%s_recall' % fold].append(recall)
                _run.info['%s_f1' % fold].append(f1)
            _run.info['time'].append(total_time)
            _run.info['epochs'].append(epoch)
            if epoch > n_epochs:
                break
            t0 = time.clock()
        data = make_data(batch,
                         augment=augment,
                         unk_idx=unk_idx,
                         singleton_idx=singleton_idx)
        model.train()
        model.zero_grad()
        loss = compute_loss(model, data, objective, loss_function)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
        optimizer.step()
        last_epoch = epoch
    dump_model(model, 'model_final.pt')

    return _run.info['test_f1'][-1]
