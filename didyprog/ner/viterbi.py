import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


class HardMaxOp:
    @staticmethod
    def max(X):
        M, _ = torch.max(X, dim=2, keepdim=True)
        A = (M == X).float()
        A = A / torch.sum(A, dim=2, keepdim=True)

        return M.squeeze(), A.squeeze()

    @staticmethod
    def hessian_product(P, Z):
        return torch.zeros_like(Z)


class SoftMaxOp:
    @staticmethod
    def max(X):
        M, _ = torch.max(X, dim=2)
        X = X - M[:, :, None]
        A = torch.exp(X)
        S = torch.sum(A, dim=2)
        M = M + torch.log(S)
        A /= S[:, :, None]
        return M.squeeze(), A.squeeze()


    @staticmethod
    def hessian_product(P, Z):
        prod = P * Z
        return prod - P * torch.sum(prod, dim=2, keepdim=True)


class SparseMaxOp:
    @staticmethod
    def max(X):
        seq_len, n_batch, n_states = X.shape
        X_sorted, _ = torch.sort(X, dim=2, descending=True)
        cssv = torch.cumsum(X_sorted, dim=2) - 1
        ind = X.new(n_states)
        for i in range(n_states):
            ind[i] = i + 1
        cond = X_sorted - cssv / ind > 0
        rho = cond.long().sum(dim=2)
        cssv = cssv.view(-1, n_states)
        rho = rho.view(-1)

        tau = (torch.gather(cssv, dim=1, index=rho[:, None] - 1)[:, 0]
               / rho.type(X.type()))
        tau = tau.view(seq_len, n_batch)
        A = torch.clamp(X - tau[:, :, None], min=0)
        # A /= A.sum(dim=2, keepdim=True)

        M = torch.sum(A * (X - .5 * A), dim=2)

        return M.squeeze(), A.squeeze()

    @staticmethod
    def hessian_product(P, Z):
        S = (P > 0).type(Z.type())
        support = torch.sum(S, dim=2, keepdim=True)
        prod = S * Z
        return prod - S * torch.sum(prod, dim=2, keepdim=True) / support


operators = {'softmax': SoftMaxOp, 'sparsemax': SparseMaxOp,
             'hardmax': HardMaxOp}


def _topological_loop(theta, batch_sizes, operator='softmax', adjoint=False,
                      Q=None, Qt=None):
    operator = operators[operator]
    new = theta.new
    B = batch_sizes[0].item()
    T = len(batch_sizes)
    L, S, _ = theta.size()
    if adjoint:
        Qd = new(L + B, S, S).zero_()
        Qtd = new(B, S).zero_()
        Vd = new(L + B, S).zero_()
        Vdt = new(B).zero_()
    else:
        Q = new(L + B, S, S).zero_()
        Qt = new(B, S).zero_()
        V = new(L + B, S).zero_()
        Vt = new(B).zero_()

    left = B
    term_right = B
    prev_length = B

    for t in range(T + 1):
        if t == T:
            cur_length = 0
        else:
            cur_length = batch_sizes[t]
        right = left + cur_length
        prev_left = left - prev_length
        prev_cut = right - prev_length
        len_term = prev_length - cur_length
        if cur_length != 0:
            # -B account for padding
            if adjoint:
                M = (theta[left - B:right - B]
                     + Vd[prev_left:prev_cut][:, None, :])
                Vd[left:right] = torch.sum(Q[left:right] * M, dim=2)
                Qd[left:right] = operator.hessian_product(Q[left:right], M)
            else:
                M = (theta[left - B:right - B]
                     + V[prev_left:prev_cut][:, None, :])
                V[left:right], Q[left:right] = operator.max(M)
        term_left = term_right - len_term
        if len_term != 0:
            if adjoint:
                M = Vd[prev_cut:left]
                Vdt[term_left:term_right] = torch.sum(
                    Qt[term_left:term_right] * M)
                Qtd[term_left:term_right] = operator.hessian_product(
                    Qt[term_left:term_right][:, None, :], M[:, None, :])[:, 0]
            else:
                M = V[prev_cut:left]
                Vt[term_left:term_right], Qt[term_left:term_right] \
                    = operator.max(M[:, None, :])
        term_right = term_left
        left = right
        prev_length = cur_length
    if adjoint:
        return Vdt, Qd, Qtd
    else:
        return Vt, Q, Qt


def _reverse_loop(Q, Qt, Ut, batch_sizes, adjoint=False,
                  U=None, Qd=None, Qdt=None):
    new = Q.new

    B = batch_sizes[0].item()
    T = len(batch_sizes)
    L, S, _ = Q.size()
    L = L - B

    if adjoint:
        Ed = new(L, S, S).zero_()
        Ud = new(L + B, S).zero_()
        Udt = new(B).zero_()
    else:
        E = new(L, S, S).zero_()
        U = new(L + B, S).zero_()
        # Ut = Ut

    right = L + B
    term_left = 0
    prev_length = 0
    off_right = L

    for t in reversed(range(-1, T)):
        if t == -1:
            cur_length = B
        else:
            cur_length = batch_sizes[t]
        left = right - cur_length
        off_left = off_right - prev_length
        cut = left + prev_length
        len_term = cur_length - prev_length
        if prev_length != 0:
            prev_left, prev_cut = right, right + prev_length
            if adjoint:
                Ed[off_left:off_right] = (Q[prev_left:prev_cut] *
                                          Ud[prev_left:prev_cut][:, :, None] +
                                          Qd[prev_left:prev_cut] *
                                          U[prev_left:prev_cut][:, :, None])
                Ud[left:cut] = torch.sum(Ed[off_left:off_right], dim=1)
            else:
                E[off_left:off_right] = (Q[prev_left:prev_cut] *
                                         U[prev_left:prev_cut][:, :, None])
                U[left:cut] = torch.sum(E[off_left:off_right], dim=1)
        term_right = term_left + len_term
        if len_term > 0:
            if adjoint:
                Ud[cut:right] = (Qt[term_left:term_right] *
                                 Udt[term_left:term_right][:, None] +
                                 Qdt[term_left:term_right] *
                                 Ut[term_left:term_right][:, None])

            else:
                U[cut:right] = (Qt[term_left:term_right]
                                * Ut[term_left:term_right][:, None])
        term_left = term_right
        right = left
        off_right = off_left
        prev_length = cur_length
    if not adjoint:
        return E, U, Ut
    else:
        return Ed, Ud, Udt


class ViterbiFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, batch_sizes, operator):
        Vt, Q, Qt = _topological_loop(theta, batch_sizes,
                                      operator=operator, adjoint=False)
        ctx.save_for_backward(theta, Q, Qt)
        ctx.others = batch_sizes, operator
        return Vt

    @staticmethod
    def backward(ctx, M):
        theta, Q, Qt = ctx.saved_tensors
        batch_sizes, operator = ctx.others
        return ViterbiFunctionBackward.apply(theta,
                                             M, Q, Qt, batch_sizes,
                                             operator), None, None


class ViterbiFunctionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, M, Q, Qt, batch_sizes, operator):
        E, U, Ut = _reverse_loop(Q, Qt, M, batch_sizes,
                                 adjoint=False)
        ctx.save_for_backward(Q, Qt, U, Ut)
        ctx.others = batch_sizes, operator
        return E

    @staticmethod
    def backward(ctx, Z):
        Q, Qt, U, Ut = ctx.saved_tensors
        batch_sizes, operator = ctx.others
        Vdt, Qd, Qdt = _topological_loop(Z, batch_sizes,
                                         operator=operator,
                                         adjoint=True,
                                         Q=Q, Qt=Qt)
        Ed, _, _ = _reverse_loop(Q, Qt, Ut, batch_sizes,
                                 adjoint=True,
                                 Qd=Qd, Qdt=Qdt,
                                 U=U)
        return Ed, Vdt, None, None, None, None


class PackedViterbi(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta):
        return ViterbiFunction.apply(theta.data, theta.batch_sizes,
                                     self.operator)

    def decode(self, theta):
        """Shortcut for doing inference
        """
        data, batch_sizes = theta
        with torch.enable_grad():
            data.requires_grad_()
            nll = self.forward(theta)
            v = torch.sum(nll)
            v_grad, = torch.autograd.grad(v, (data,), create_graph=True)
        return PackedSequence(v_grad, batch_sizes)


class Viterbi(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.packed_viterbi = PackedViterbi(operator=operator)

    def _pack(self, theta, lengths):
        T, B, S, _ = theta.shape
        if lengths is None:
            data = theta.view(T * B, S, S)
            batch_sizes = torch.LongTensor(T, device=theta.device).fill_(B)
        else:
            data, batch_sizes = pack_padded_sequence(theta, lengths)
        return PackedSequence(data, batch_sizes)

    def forward(self, theta, lengths=None):
        return self.packed_viterbi(self._pack(theta, lengths))

    def decode(self, theta, lengths=None):
        return self.packed_viterbi.decode(self._pack(theta, lengths))
