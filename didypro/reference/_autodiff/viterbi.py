"""
Pytorch core for Viterbi and ViterbiGrad layers.
"""

import torch
from didypro.reference._autodiff.local import operators

from torch import nn
from torch.autograd import grad


class Viterbi(nn.Module):
    """
        Value layer based on Viterbi algorithm.
    """
    def __init__(self, operator: str = 'hardmax') -> None:
        """
        :param operator: str in {'hardmax', 'softmax', 'sparsemax'}
        """
        super().__init__()
        self.operator = operators[operator]

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        :param theta: torch.Tensor, shape=(T, B, S, S)
            Potential tensor on which to apply the Viterbi alg.
        :return: values: torch.Tensor, shape=(B)
            Output of the Viterbi loop for each sample
        """
        T, B, S, _ = theta.shape
        V = theta.new_zeros(B, S)
        for t in range(T):
            V = self.operator(theta[t] + V[:, None, :])
        v = self.operator(V[:, None, :])[:, 0]
        return v

    def decode(self, theta):
        with torch.enable_grad():
            theta.requires_grad_()
            nll = self.forward(theta)
            v = torch.sum(nll)
            theta_grad, = torch.autograd.grad(v, (theta,), create_graph=True)
        return theta_grad


class ViterbiGrad(nn.Module):
    """
        Inference layer based on Viterbi algorithm.

        Derived from the value layer using AD from Pytorch.
    """
    def __init__(self, operator: str = 'hardmax') -> None:
        """
        :param operator: str in {'hardmax', 'softmax', 'sparsemax'}
        """
        super().__init__()
        self.viterbi = Viterbi(operator=operator)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        :param theta: torch.Tensor, shape=(T, S, S)
            Potential tensor on which to apply the Viterbi + backtracking alg.
        :return: y: torch.Tensor, shape=(T, B, S, S)
            Backtracking of the Viterbi loop for each sample
        """
        theta.requires_grad_()
        nll = self.viterbi(theta)
        v = torch.sum(nll)
        v_grad, = grad(v, (theta,), create_graph=True)
        return v_grad