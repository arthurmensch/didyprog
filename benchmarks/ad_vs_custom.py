import time

import torch
from didypro.modules.potential import LinearPotential
from didypro.modules.viterbi import PackedViterbi
from didypro.reference._autodiff.viterbi import Viterbi as ViterbiAD
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


length = 100
batch_size = 256
n_targets = 32
n_features = 100
n_trials = 1
gpu = False

X = torch.FloatTensor(length, batch_size, n_features).uniform_()

viterbi_ad = ViterbiAD(operator='softmax')
viterbi = PackedViterbi(operator='softmax')
linear_potential = LinearPotential(n_features, n_targets)
theta = linear_potential(X)

lengths = torch.LongTensor(batch_size).fill_(length)

theta = theta.detach()
theta.requires_grad_()

devices = [torch.device('cpu')]
if torch.cuda.is_available() and gpu:
    devices.append(torch.device('cuda:0'))
funcs = {'custom': viterbi, 'ad': viterbi_ad}

for device in devices:

    print('Device: %s +++++++++++++++++++++++++' % device)
    print('Forward timing')

    for name, func in funcs.items():
        if name == 'custom':
            this_theta = pack_padded_sequence(theta, lengths)
        else:
            this_theta = theta
        print('%s : ------------' % name)
        total = 0
        for i in range(n_trials):
            t0 = time.clock()
            with torch.no_grad():
                value = torch.sum(func(this_theta))
            total += time.clock() - t0
            print('Value', value.item())
        total /= n_trials
        print('Time %.4f s' % total)

    print('###########################')

    theta.requires_grad_()
    print('Forward + backward timing')
    for name, func in funcs.items():
        if name == 'custom':
            this_theta = pack_padded_sequence(theta, lengths)
        else:
            this_theta = theta
        print('%s : ------------' % name)
        total = 0
        for i in range(n_trials):
            t0 = time.clock()
            if theta.grad is not None:
                theta.grad.zero_()
            value = torch.sum(func(this_theta))
            with torch.autograd.profiler.profile() as prof:
                value.backward()
            total += time.clock() - t0
            print('Value', value.item())
            print('|g|', torch.sum(torch.abs(theta.grad)).item())
            prof.export_chrome_trace('prof_%s.txt' % name)
        total /= n_trials
        print('Time %.4f s' % total)

    print('###########################')

    print('Forward + double backward timing')
    z = torch.randn_like(theta)
    for name, func in funcs.items():
        if name == 'custom':
            this_theta = pack_padded_sequence(theta, lengths)
        else:
            this_theta = theta
        print('%s : ------------' % name)
        total = 0
        for i in range(n_trials):
            t0 = time.clock()
            if theta.grad is not None:
                theta.grad.zero_()
            value = torch.sum(func(this_theta))
            g, = torch.autograd.grad(value, (theta,), create_graph=True)
            s = torch.sum(g * z)
            with torch.autograd.profiler.profile() as prof:
                s.backward()
            total += time.clock() - total
            print('Value', value.item())
            print('|g|', torch.sum(torch.abs(theta.grad)).item())
            prof.export_chrome_trace('prof_double_%s.txt' % name)
        total /= n_trials
        print('Time %.4f s' % total)
