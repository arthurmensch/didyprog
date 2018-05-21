import time

import torch
from didypro.modules.potential import LinearPotential
from didypro.modules.viterbi import Viterbi
from didypro.reference._autodiff.viterbi import Viterbi as ViterbiAD

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


length = 100
batch_size = 256
n_targets = 32
n_features = 100
n_trials = 1
gpu = True
operator = 'sparsemax'

X = torch.FloatTensor(length, batch_size, n_features).uniform_()

viterbi_ad = ViterbiAD(operator=operator)
viterbi = Viterbi(operator=operator)
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
