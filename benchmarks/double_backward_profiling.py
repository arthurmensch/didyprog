import torch
from didyprog.ner.potential import LinearPotential
from didyprog.ner.viterbi import Viterbi

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


length = 100
batch_size = 256
n_targets = 32
n_features = 100
gpu = True
operator = 'sparsemax'

if gpu and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

X = torch.FloatTensor(length, batch_size, n_features).uniform_()

viterbi = Viterbi(operator=operator)
linear_potential = LinearPotential(n_features, n_targets)
theta = linear_potential(X)
theta = theta.detach()
theta = theta.to(device)
theta.requires_grad_()
z = torch.randn_like(theta)

value = torch.sum(viterbi(theta))
g, = torch.autograd.grad(value, (theta,), create_graph=True)
s = torch.sum(g * z)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    s.backward()
print('Value', value.item())
print('|g|', torch.sum(torch.abs(theta.grad)).item())
prof.export_chrome_trace('prof.txt')
