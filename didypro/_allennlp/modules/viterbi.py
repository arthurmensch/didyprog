"""
Viterbi and Viterbi inference with autograd.
"""

import torch
from torch.autograd import Variable


def viterbi(theta: torch.Tensor,
            mask: torch.Tensor = None,
            operator: str = 'softmax') -> torch.Tensor:
    """

    :param theta: torch.Tensor, shape=(T, B, S, S)
        Potential tensor on which to apply the Viterbi alg.
    :param mask:
    :param operator:
    :return: values: torch.Tensor, shape=(B)
        Output of the Viterbi loop for each sample
    """
    operator = operators[operator]
    T, B, S, _ = theta.shape
    V = Variable(theta.data.new(B, S).zero_())
    if mask is None:
        mask = Variable(theta.data.new(T, B).fill_(1.))
    for t in range(0, T):
        V = (mask[t][:, None] * operator(theta[t] + V[:, None, :])
             + (1 - mask[t])[:, None] * V)
    v = operator(V[:, None, :])[:, 0]
    return v


def viterbi_decode(theta, mask: torch.Tensor = None,
                   operator: str = 'hardmax'):
    # theta = theta.detach()
    # theta.requires_grad = True
    nll = viterbi(theta, mask, operator=operator)
    v = torch.sum(nll)
    theta_grad, = torch.autograd.grad(v, (theta,), create_graph=True)
    return theta_grad


def softmax(X):
    """
    Entropy-smoothed max, a.k.a. logsumexp.

    Solves $max_{p \in \Delta^d} <x, p> - \sum_{i=1}^d p_i \log(p_i)$ along
    dim=2.

    :param x: torch.Tensor, shape = (b, n, m)
        Vector to project

    :return: torch.Tensor, shape = (b, n)
        Projected vector
    """
    M, _ = torch.max(X, dim=2)
    X = X - M[:, :, None]
    S = torch.sum(torch.exp(X), dim=2)
    M = M + torch.log(S)
    return M


def sparsemax(X):
    """
    L2-smoothed max, a.k.a. sparsemax value.

    Solves $max_{p \in \Delta^d} <x, p> - \sum_{i=1}^d p_i^2$ along
    dim=2.

    :param x: torch.Tensor, shape = (b, n, m)
        Vector to project

    :return: torch.Tensor, shape = (b, n)
        Projected vector
    """
    seq_len, n_batch, n_states = X.shape
    X_sorted, _ = torch.sort(X, dim=2, descending=True)
    cssv = torch.cumsum(X_sorted, dim=2) - 1
    ind = X.new_empty(n_states)
    for i in range(n_states):
        ind[i] = i + 1
    cond = X_sorted - cssv / ind > 0
    rho = cond.long().sum(dim=2)
    cssv = cssv.view(-1, n_states)
    rho = rho.view(-1)
    tau = torch.gather(cssv, dim=1,
                       index=rho[:, None] - 1)[:, 0] / rho.type(X.type())
    tau = tau.view(seq_len, n_batch)
    A = torch.clamp(X - tau[:, :, None], min=0)
    A /= A.sum(dim=2, keepdim=True)

    M = torch.sum(A * (X - .5 * A), dim=2)
    return M


def hardmax(X):
    """
    L2-smoothed max, a.k.a. sparsemax value.

    Solves $max_{p \in \Delta^d} <x, p> along axis=2.

    :param x: torch.Tensor, shape = (b, n, m)
        Vector to project

    :return: torch.Tensor, shape = (b, n)
        Projected vector
    """
    M, _ = torch.max(X, dim=2)
    return M


operators = {'hardmax': hardmax, 'softmax': softmax, 'sparsemax': sparsemax}