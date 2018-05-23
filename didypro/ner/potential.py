import torch

import torch.nn as nn

from torch.nn import Parameter


class LinearPotential(torch.nn.Module):
    def __init__(self, n_features, n_states, init_idx=None, eos_idx=None):
        super(LinearPotential, self).__init__()

        self.transition = Parameter(torch.zeros((n_states, n_states)))
        self.init_idx = init_idx
        self.eos_idx = eos_idx

        self.weight = Parameter(torch.zeros((n_features, n_states)))
        self.bias = Parameter(torch.zeros(n_states))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.transition)
        self.bias.data.fill_(0.)

    def forward(self, X):
        n_states = self.transition.shape[0]
        batch_size = X.shape[0]
        unary_potentials = (torch.matmul(X, self.weight[None, :, :])
                            + self.bias[None, None, :])
        potentials = (unary_potentials[:, :, :, None]
                      + self.transition[None, None, :, :])
        if self.init_idx is not None:
            # Non emitting first state
            potentials[:, 0, :, :] = 0
        else:
            potentials[:, 0, :, :] = unary_potentials[:, 0, :, None].expand(-1, -1, n_states)
        if self.eos_idx is not None:
            # Non emitting last state
            potentials[:, -1, :, :] = self.transition[None, :, :].expand(batch_size, -1, -1)
        return potentials