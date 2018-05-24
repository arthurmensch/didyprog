import numpy as np
import pytest
from didypro.utils import make_data
from scipy.optimize import check_grad

from didypro.reference.viterbi import viterbi_grad, viterbi_value, \
    viterbi_hessian_prod


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_viterbi(operator):
    states, emissions, theta = make_data(100)
    viterbi, grad, _, _ = viterbi_grad(theta, operator=operator)
    decoded = np.argmax(grad.sum(axis=2), axis=1)
    assert np.all(decoded == states)


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_viterbi_grad(operator):
    states, emissions, theta = make_data()
    theta /= 100

    def func(X):
        X = X.reshape(theta.shape)
        return viterbi_value(X, operator=operator)

    def grad(X):
        X = X.reshape(theta.shape)
        _, grad, _, _ = viterbi_grad(X, operator=operator)
        return grad.ravel()

    # check_grad does not work with ndarray of dim > 2
    err = check_grad(func, grad, theta.ravel())
    if operator == 'sparsemax':
        assert err < 1e-4
    else:
        assert err < 1e-6


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_viterbi_hessian(operator):
    states, emissions, theta = make_data()

    theta /= 100
    Z = np.random.randn(*theta.shape)

    def func(X):
        X = X.reshape(theta.shape)
        _, grad, _, _ = viterbi_grad(X, operator=operator)
        return np.sum(grad * Z)

    def grad(X):
        X = X.reshape(theta.shape)
        _, H = viterbi_hessian_prod(X, Z, operator=operator)
        return H.ravel()

    # check_grad does not work with ndarray of dim > 2
    err = check_grad(func, grad, theta.ravel())
    if operator == 'sparsemax':
        assert err < 1e-4
    else:
        assert err < 1e-6
