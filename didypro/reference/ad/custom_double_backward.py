import torch

import torch.nn as nn


class CubeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)
        return X ** 3

    @staticmethod
    def backward(ctx, M):
        X, = ctx.saved_tensors
        return CubeFunctionBackward.apply(X, M)


class CubeFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, M):
        ctx.save_for_backward(X)
        return M * 3 * X ** 2

    @staticmethod
    def backward(ctx, V):
        X, = ctx.saved_tensors
        return V * 6 * X, None


class Cube(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return CubeFunction.apply(X)

torch.manual_seed(0)
X = torch.Tensor([[1., 2.], [3., 4.]])
X.requires_grad_()
print(X)
cube = Cube()

Y = cube(X)
S = torch.sum(Y)
S.backward()
print(X.grad)

X.grad.zero_()
X.requires_grad_()
Y = cube(X)
S = torch.sum(Y)
G, = torch.autograd.grad(S, (X, ), create_graph=True)
print(G)
S = G[0, 0]
S.backward()
print(X.grad)