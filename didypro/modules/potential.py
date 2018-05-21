import torch

import torch.nn as nn

from torch.nn import Parameter


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
        unary_potentials = (torch.matmul(X, self.weight[None, :, :])
                            + self.bias[None, None, :])
        potentials = (unary_potentials[:, :, :, None]
                      + self.transition[None, None, :, :])
        return potentials
