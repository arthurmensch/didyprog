import torch

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

        tau = torch.gather(cssv, dim=1,
                           index=rho[:, None] - 1)[:, 0] / rho.type(X.type())
        tau = tau.view(seq_len, n_batch)
        A = torch.clamp(X - tau[:, :, None], min=0)

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
