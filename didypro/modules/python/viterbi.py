import torch

from torch import nn
from torch.autograd import grad

# class SoftMax(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, X: torch.Tensor) -> torch.Tensor:
#         M, _ = torch.max(X, dim=2)
#         X = X - M[:, :, None]
#         A = torch.exp(X)
#         S = torch.sum(A, dim=2)
#         M = M + torch.log(S)
#         A = A / S[:, :, None]
#         ctx.save_for_backward(A)
#         return M
#
#     @staticmethod
#     def backward(ctx, M):
#         A, = ctx.saved_tensors
#         return M[:, :, None] * A
#
#
# class SparseMax(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, X: torch.Tensor) -> torch.Tensor:
#         seq_len, n_batch, n_states = X.shape
#         X_sorted, _ = torch.sort(X, dim=2, descending=True)
#         cssv = torch.cumsum(X_sorted, dim=2) - 1
#         ind = X.new_empty(n_states)
#         for i in range(n_states):
#             ind[i] = i + 1
#         cond = X_sorted - cssv / ind > 0
#         rho = cond.long().sum(dim=2)
#         cssv = cssv.view(-1, n_states)
#         rho = rho.view(-1)
#         tau = torch.gather(cssv, dim=1,
#                            index=rho[:, None] - 1)[:, 0] / rho.type(X.type())
#         tau = tau.view(seq_len, n_batch)
#         A = torch.clamp(X - tau[:, :, None], min=0)
#
#         M = torch.sum(A * (X - .5 * A), dim=2)
#
#         ctx.save_for_backward(A)
#         return M
#
#     @staticmethod
#     def backward(ctx, M):
#         A, = ctx.saved_tensors
#         return M[:, :, None] * A
#
#
# class HardMax(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, X: torch.Tensor) -> torch.Tensor:
#         M, idx = torch.max(X, dim=2)
#         A = torch.zeros_like(X)
#         A.scatter_(2, idx[:, :, None], 1.)
#         ctx.save_for_backward(A)
#         return M
#
#
#     @staticmethod
#     def backward(ctx, M):
#         A, = ctx.saved_tensors
#         return M[:, :, None] * A
#
#

def softmax(X):
    M, _ = torch.max(X, dim=2)
    X = X - M[:, :, None]
    S = torch.sum(torch.exp(X), dim=2)
    M = M + torch.log(S)
    return M


def sparsemax(X):
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

    M = torch.sum(A * (X - .5 * A), dim=2)
    return M


def hardmax(X):
    M, _ = torch.max(X, dim=2)
    return M


class Viterbi(nn.Module):
    def __init__(self, operator: str = 'hardmax') -> None:
        super().__init__()
        self.operator = operators[operator]

    def forward(self, theta):
        T, B, S, _ = theta.shape
        V = theta.new_zeros(B, S)
        V[:, 1:].fill_(-1e10)
        for t in range(T):
            V = self.operator(theta[t] + V[:, None, :])
        v = self.operator(V[:, None, :])[:, 0]
        return v


class ViterbiGrad(nn.Module):
    def __init__(self, operator: str = 'hardmax') -> None:
        super().__init__()
        self.viterbi = Viterbi(operator=operator)

    def forward(self, theta):
        nll = self.viterbi(theta)
        theta.requires_grad_()
        v = torch.sum(nll)
        v_grad, = grad(v, (theta,), create_graph=True)
        return v_grad


operators = {'hardmax': hardmax, 'softmax': softmax, 'sparsemax': sparsemax}
