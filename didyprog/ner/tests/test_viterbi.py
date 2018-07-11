import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from didyprog.ner.viterbi import Viterbi, PackedViterbi
from didyprog.utils import make_data


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_packed_viterbi(operator):
    states, emissions, theta = make_data(10)

    theta = torch.from_numpy(theta)
    theta = theta[:, None, :, :]
    theta = theta.repeat((1, 2, 1, 1))
    theta.requires_grad_()
    W = pack_padded_sequence(theta, [10, 10])

    viterbi = PackedViterbi(operator)
    v = viterbi(W)
    s = v.sum()
    s.backward()
    decoded = torch.argmax(theta.grad[:, 0].sum(dim=2), dim=1).numpy()
    assert np.all(decoded == states)


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_viterbi(operator):
    states, emissions, theta = make_data(10)

    theta = torch.from_numpy(theta)
    theta.requires_grad_()
    W = theta[:, None, :, :]

    viterbi = Viterbi(operator)
    v = viterbi(W)
    s = v.sum()
    s.backward()
    decoded = torch.argmax(theta.grad.sum(dim=2), dim=1).numpy()
    assert np.all(decoded == states)


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_viterbi_two_lengths(operator):
    states1, emissions1, theta1 = make_data(10)
    states2, emissions2, theta2 = make_data(5)
    lengths = torch.LongTensor([10, 5])

    theta1 = torch.from_numpy(theta1)
    theta2 = torch.from_numpy(theta2)

    theta1.requires_grad_()
    theta2.requires_grad_()
    W = pad_sequence([theta1, theta2])

    viterbi = Viterbi(operator)
    v = viterbi(W, lengths=lengths)
    s = v.sum()
    s.backward()
    decoded1 = torch.argmax(theta1.grad.sum(dim=2), dim=1).numpy()
    decoded2 = torch.argmax(theta2.grad.sum(dim=2), dim=1).numpy()
    assert np.all(decoded1 == states1)
    assert np.all(decoded2 == states2)


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_grad_viterbi(operator):
    states, emissions, theta = make_data(10)

    theta = torch.from_numpy(theta)
    theta = theta[:, None, :, :]
    theta.requires_grad_()

    viterbi = Viterbi(operator)
    gradcheck(viterbi, (theta, ))


@pytest.mark.parametrize("operator", ['softmax', 'sparsemax'])
def test_grad_grad_viterbi(operator):
    states, emissions, theta = make_data(10)

    theta = torch.from_numpy(theta)
    theta = theta[:, None, :, :]
    theta.requires_grad_()

    viterbi = Viterbi(operator)
    gradgradcheck(viterbi, (theta, ))


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_grad_hessian_viterbi_two_samples(operator):
    states1, emissions1, theta1 = make_data(10)
    states2, emissions2, theta2 = make_data(5)
    lengths = torch.LongTensor([10, 5])

    theta1 = torch.from_numpy(theta1)
    theta2 = torch.from_numpy(theta2)

    theta1.requires_grad_()
    theta2.requires_grad_()

    viterbi = Viterbi(operator)

    def func(theta1_, theta2_):
        W = pad_sequence([theta1_, theta2_])
        return viterbi(W, lengths)

    gradcheck(func, (theta1, theta2))
    gradgradcheck(func, (theta1, theta2))


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_hessian_viterbi(operator):
    torch.manual_seed(0)

    states, emissions, theta = make_data(10)
    theta /= 100

    theta = torch.from_numpy(theta)
    theta = theta[:, None, :, :]
    theta.requires_grad_()

    viterbi = Viterbi(operator)
    ll = viterbi(theta)
    g, = torch.autograd.grad(ll, (theta, ), create_graph=True)
    z = torch.randn_like(g)
    s = torch.sum(g * z)
    s.backward()

    assert theta.grad.shape == (10, 1, 3, 3)
