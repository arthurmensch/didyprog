import torch

import torch.nn as nn


class ViterbiFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)
        return X ** 3

    @staticmethod
    def backward(ctx, M):
        X, = ctx.saved_tensors
        return ViterbiFunctionBackward.apply(X, M)


class ViterbiFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, M):
        ctx.save_for_backward(X)
        return 3 * M * X ** 2

    @staticmethod
    def backward(ctx, V):
        X, = ctx.saved_tensors
        return 6 * V * X, 3 * V * X ** 2


class Cube(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return CubeFunction.apply(X)