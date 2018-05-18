import numpy as np
import pytest

import torch
from didypro.modules.python.viterbi import ViterbiGrad, Viterbi

from didypro.reference.tests.test_viterbi import make_data
from didypro.reference.viterbi import viterbi_grad, viterbi_hessian_prod


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_viterbi(operator):
    states, emissions, theta = make_data(3)
    nll_ref, grad_ref, _, _ = viterbi_grad(theta, operator=operator)

    theta = torch.from_numpy(theta[:, None, :, :])
    theta.requires_grad_()
    viterbi = Viterbi(operator)
    nll = viterbi(theta)
    nll = nll.sum()

    np.testing.assert_almost_equal(nll.item(), nll_ref)

    nll.backward()

    grad = theta.grad[:, 0].numpy()
    np.testing.assert_array_almost_equal(grad, grad_ref)


@pytest.mark.parametrize("operator", ['softmax', 'sparsemax'])
def test_viterbi_grad(operator):
    states, emissions, theta = make_data(10)
    theta = theta / 100

    Z = np.zeros_like(theta)
    Z[1, 2, 1] = 1
    #
    # rng = np.random.RandomState(0)
    # Z = rng.randn(*theta.shape)
    #
    _, hessian_prod_ref = viterbi_hessian_prod(theta, Z,
                                               operator=operator)

    Z = torch.from_numpy(Z[:, None, :, :])
    theta = torch.from_numpy(theta[:, None, :, :])
    theta.requires_grad_()
    viterbi_grad = ViterbiGrad(operator)
    v_grad = viterbi_grad(theta)

    v_h = torch.sum(Z * v_grad)
    v_h.backward()
    hessian_prod = theta.grad[:, 0].numpy()

    assert np.testing.assert_array_almost_equal(hessian_prod,
                                                hessian_prod_ref)
