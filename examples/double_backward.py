import torch

import torch.nn as nn
from torch.autograd.gradcheck import gradgradcheck, gradcheck


class CubeFunction(torch.autograd.Function):
    """
    Dummy activation function x -> x ** 3
    """
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
        ctx.save_for_backward(X, M)
        return M * 3 * X ** 2

    @staticmethod
    def backward(ctx, V):
        X, M = ctx.saved_tensors
        return V * 6 * X * M, V * 3 * X ** 2


class Cube(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return CubeFunction.apply(X)


class TrueCube(nn.Module):
    """
    Pytorch x -> x ** 3 is twice differentiable already, we use it for reference.
    """
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X ** 3

torch.manual_seed(0)

X = torch.Tensor([3.])
X.requires_grad_()
print('x:', X)
cube = Cube()

Y = cube(X)
print('f(x):', Y)
S = torch.sum(Y)
S.backward()
print('<Grad (f)(x), 1>:', X.grad)

X.grad.zero_()
X.requires_grad_()
Y = cube(X)
S = torch.sum(Y)
G, = torch.autograd.grad(S, (X, ), create_graph=True)
S = G.sum()
S.backward()
print('Grad^2 (f) 1:', X.grad)

X.grad.zero_()
gradcheck(cube, (X, ), eps=1e-4, atol=1e-2)
X.grad.zero_()
gradgradcheck(cube, (X, ), eps=1e-4, atol=1e-2)
