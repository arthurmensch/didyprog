import torch
import torch.nn as nn

from didyprog.ner.core.local import operators


def _topological_loop(theta, mask, operator='softmax', adjoint=False,
                      Q=None):
    operator = operators[operator]
    new = theta.new
    B, T, S, _ = theta.size()
    if adjoint:
        Qd = new(B, T + 2, S, S).zero_()
        Vd = new(B, T + 1, S).zero_()
    else:
        Q = new(B, T + 2, S, S).zero_()
        V = new(B, T + 1, S).zero_()

    for t in range(1, T + 1):
        if adjoint:
            M = (theta[:, t - 1] + V[:, t - 1][:, None, :])
            Vd[:, t] = torch.sum(Q[:, t] * M, dim=2) * mask[:, t][:, None]
            Qd[:, t] = operator.hessian_product(Q[:, t], M) * mask[:, t][:,
                                                              None, None]
            Vd[:, t] = (Vd[:, t] * mask[:, t][:, None] +
                        (1 - mask[:, t][:, None]) * Vd[:, t - 1])
            Qd[:, t] = Qd[:, t] * mask[:, t][:, None, None]
        else:
            V[:, t], Q[:, t] = operator.max(theta[:, t - 1]
                                            + V[:, t - 1][:, None, :])
            V[:, t] = (V[:, t] * mask[:, t][:, None] +
                       (1 - mask[:, t][:, None]) * V[:, t - 1])
            Q[:, t] = Q[:, t] * mask[:, t][:, None, None]
    if adjoint:
        v, Q[:, T + 1, 0] = operator.max(V[:, T][:, None, :])
    else:
        M = V[:, T][:, None, :]
        vd = torch.sum(Q[:, T + 1, 0] * M, dim=2)
        Qd[:, T + 1, 0] = operator.hessian_product(Q[:, T + 1, 0], M)

    if adjoint:
        return vd, Qd
    else:
        return v, Q


def _reverse_loop(Q, mask, M=None, adjoint=False, U=None, Qd=None):
    new = Q.new
    B, T, S, _ = Q.size()
    T = T - 2
    if adjoint:
        Ed = new(B, T + 1, S, S).zero_()
        Ud = new(B, T + 2, S).zero_()
    else:
        E = new(B, T + 1, S, S).zero_()
        U = new(B, T + 2, S).zero_()
        U[:, T + 1, 0] = M

    for t in reversed(range(0, T + 1)):
        if adjoint:
            Ed[:, t] = (Q[t + 1] * Ud[t + 1][:, None]
                        + Qd[t + 1] * U[t + 1][:, None]) * mask[:, t][:, None, None]
            Ud[:, t] = torch.sum(Ed[:, t], dim=0)
        else:
            E[:, t] = Q[:, t + 1] * U[:, t + 1][:, None]
            U[:, t] = (torch.sum(E[:, t], dim=0) * mask[:, t][:, None]
                       + (1 - mask[:, t][:, None]) * U[:, t + 1])
    if adjoint:
        return E, U
    else:
        return Ed, Ud


class ViterbiFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, mask, operator):
        v, Q = _topological_loop(theta, mask, operator=operator, adjoint=False)
        ctx.save_for_backward(theta, mask, Q)
        ctx.operator = operator
        return v

    @staticmethod
    def backward(ctx, M):
        theta, mask, Q = ctx.saved_tensors
        operator = ctx.operator
        return (ViterbiFunctionBackward.apply(theta, mask, M, Q, operator),
                None, None)


class ViterbiFunctionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, mask, M, Q, operator):
        E, U = _reverse_loop(Q, M, adjoint=False)
        ctx.save_for_backward(mask, Q, E, U)
        ctx.operator = operator
        return E[:-1]

    @staticmethod
    def backward(ctx, Z):
        mask, Q, E, U = ctx.saved_tensors
        batch_sizes, operator = ctx.others
        vd, Qd = _topological_loop(Z, mask, operator=operator,
                                   adjoint=True, Q=Q)
        Ed, Ud = _reverse_loop(Q, adjoint=True, Qd=Qd, U=U)
        return Ed[:-1], None, vd, None, None


class MaskedViterbi(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, mask):
        return ViterbiFunction.apply(theta, mask,
                                     self.operator)

    def decode(self, theta, mask):
        """Shortcut for doing inference
        """
        with torch.enable_grad():
            theta.requires_grad_()
            nll = self.forward(theta, mask)
            v = torch.sum(nll)
            theta_grad, = torch.autograd.grad(v, (theta,), create_graph=True)
        return theta_grad
#
#
# class Viterbi(nn.Module):
#     def __init__(self, operator):
#         super().__init__()
#         self.packed_viterbi = PackedViterbi(operator=operator)
#
#     def _pack(self, theta, lengths):
#         T, B, S, _ = theta.shape
#         if lengths is None:
#             data = theta.view(T * B, S, S)
#             batch_sizes = torch.LongTensor(T, device=theta.device).fill_(B)
#         else:
#             data, batch_sizes = pack_padded_sequence(theta, lengths)
#         return PackedSequence(data, batch_sizes)
#
#     def forward(self, theta, lengths=None):
#         return self.packed_viterbi(self._pack(theta, lengths))
#
#     def decode(self, theta, lengths=None):
#         return self.packed_viterbi.decode(self._pack(theta, lengths))
