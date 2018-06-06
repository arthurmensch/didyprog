import time

import functools
import torch
from didypro.ner.potential import LinearPotential
from didypro.ner.viterbi import Viterbi
from didypro._allennlp.modules.viterbi import viterbi as viterbi_ad

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


length = 100
batch_size = 32
n_targets = 16
n_features = 100
n_trials = 1
gpu = True
operator = 'sparsemax'

X = torch.FloatTensor(length, batch_size, n_features).uniform_()

viterbi = Viterbi(operator=operator)
viterbi_ad = functools.partial(viterbi_ad, operator=operator)

linear_potential = LinearPotential(n_features, n_targets)
theta_ = linear_potential(X)


devices = [torch.device('cpu')]
if torch.cuda.is_available() and gpu:
    devices.append(torch.device('cuda:0'))
funcs = {'custom': viterbi, 'ad': viterbi_ad}

for device in devices:
    print('Device: %s +++++++++++++++++++++++++' % device)
    print('Forward timing')
    theta = theta_.clone().detach()
    theta = theta.to(device)
    theta.requires_grad_()

    for name, func in funcs.items():
        print('%s : ------------' % name)
        total = 0
        for i in range(n_trials):
            t0 = time.clock()
            with torch.no_grad():
                value = torch.sum(func(theta))
            total += time.clock() - t0
            print('Value', value.item())
        total /= n_trials
        print('Time %.4f s' % total)

    print('###########################')

    print('Forward + backward timing')
    for name, func in funcs.items():
        print('%s : ------------' % name)
        total = 0
        for i in range(n_trials):
            t0 = time.clock()
            if theta.grad is not None:
                theta.grad.zero_()
            value = torch.sum(func(theta))
            value.backward()
            total += time.clock() - t0
            print('Value', value.item())
            print('|g|', torch.sum(torch.abs(theta.grad)).item())
        total /= n_trials
        print('Time %.4f s' % total)

    print('###########################')

    print('Forward + double backward timing')
    z = torch.randn_like(theta)
    for name, func in funcs.items():
        print('%s : ------------' % name)
        total = 0
        for i in range(n_trials):
            t0 = time.clock()
            if theta.grad is not None:
                theta.grad.zero_()
            value = torch.sum(func(theta))
            g, = torch.autograd.grad(value, (theta,), create_graph=True)
            s = torch.sum(g * z)
            s.backward()
            total += time.clock() - t0
            print('Value', value.item())
            print('|g|', torch.sum(torch.abs(theta.grad)).item())
        total /= n_trials
        print('Time %.4f s' % total)
