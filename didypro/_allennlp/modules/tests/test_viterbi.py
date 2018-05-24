import numpy as np
import pytest

import torch
from didypro._allennlp.modules.viterbi import viterbi, viterbi_decode

from didypro.utils import make_data
from didypro.reference.viterbi import viterbi_grad, viterbi_hessian_prod


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_viterbi(operator):
    states, emissions, theta = make_data(4)
    nll_ref, grad_ref, _, _ = viterbi_grad(theta, operator=operator)

    theta = torch.from_numpy(theta[:, None, :, :])
    theta.requires_grad_()
    nll = viterbi(theta, operator=operator)
    nll = nll.sum()

    np.testing.assert_almost_equal(nll.item(), nll_ref)

    nll.backward()

    grad = theta.grad[:, 0].numpy()
    np.testing.assert_array_almost_equal(grad, grad_ref)


@pytest.mark.parametrize("operator", ['softmax', 'sparsemax'])
def test_viterbi_grad(operator):
    states, emissions, theta = make_data(4)
    theta = theta / 100

    Z = np.zeros_like(theta)
    Z[1, 2, 1] = 1

    _, hessian_prod_ref = viterbi_hessian_prod(theta, Z,
                                               operator=operator)

    Z = torch.from_numpy(Z[:, None, :, :])
    theta = torch.from_numpy(theta[:, None, :, :])
    theta.requires_grad_()
    v_grad = viterbi_decode(theta, operator=operator)

    v_h = torch.sum(Z * v_grad)
    v_h.backward()
    hessian_prod = theta.grad[:, 0].numpy()

    np.testing.assert_array_almost_equal(hessian_prod, hessian_prod_ref)
