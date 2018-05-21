import torch

import torch.nn as nn

from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence


class LinearPotential(torch.nn.Module):
    def __init__(self, n_features, n_states):
        super(LinearPotential, self).__init__()

        self.transition = Parameter(torch.zeros((n_states, n_states)))

        self.weight = Parameter(torch.zeros((n_features, n_states)))
        self.bias = Parameter(torch.zeros(n_states))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.transition)
        self.bias.data.fill_(0.)

    def forward(self, X):
        n_states = self.transition.shape[0]
        unary_potentials = (torch.matmul(X, self.weight[None, :, :])
                            + self.bias[None, None, :])
        potentials = (unary_potentials[:, :, :, None]
                      + self.transition[None, None, :, :])
        potentials[:, 0, :, :] = (unary_potentials[:, 0, :, None]
                                  .expand(-1, -1, n_states))
        return potentials


class PackedLinearPotential(torch.nn.Module):
    def __init__(self, n_features, n_states, alpha=1.):
        super(PackedLinearPotential, self).__init__()

        self.transition = Parameter(torch.zeros((n_states, n_states)))
        self.weight = Parameter(torch.zeros((n_features, n_states)))
        self.bias = Parameter(torch.zeros(n_states))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.weight)
        self.transition.data.fill_(0.)
        self.bias.data.fill_(0.)

    def forward(self, X):
        data, batch_sizes = X
        new = self.weight.data.new
        n_samples = X.shape[0]
        n_states = self.transition.shape[0]
        batch_size = batch_sizes[0]
        unary_potentials = torch.matmul(X[0], self.weight) + self.bias[None, :]
        potentials = new(n_samples, n_states, n_states)
        potentials[:batch_size] = unary_potentials[:batch_size][:, :, None]. \
            expand(-1, -1, n_states)
        if n_samples > batch_size:
            potentials[batch_size:] = (
                    unary_potentials[batch_size:]
                    [:, :, None] + self.transition[None, :, :])
        return PackedSequence(potentials, batch_sizes)
