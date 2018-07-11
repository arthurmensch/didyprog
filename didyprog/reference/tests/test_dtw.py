import numpy as np
import pytest
from didyprog.reference.dtw import dtw_value, dtw_grad, dtw_hessian_prod
from scipy.optimize import check_grad
from sklearn.metrics.pairwise import pairwise_distances


def make_data():
    rng = np.random.RandomState(0)
    m, n = 2, 2
    X = rng.randn(m, 3)
    Y = rng.randn(n, 3)
    return pairwise_distances(X, Y) / 10


def test_dtw():
    C = make_data()
    dtw = dtw_value(C, operator='hardmax')

    _, grad, _, _ = dtw_grad(C, operator='hardmax')
    _, soft_grad, _, _ = dtw_grad(C, operator='softmax')
    _, sparse_grad, _, _ = dtw_grad(C, operator='sparsemax')

    assert(dtw == np.sum(grad * C))
    assert(dtw < np.sum(soft_grad * C))
    assert(dtw < np.sum(sparse_grad * C))


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_dtw_grad(operator):
    C = make_data()

    def func(X):
        X = X.reshape(C.shape)
        return dtw_value(X, operator=operator)

    def grad(X):
        X = X.reshape(C.shape)
        _, g, _, _ = dtw_grad(X, operator=operator)
        return g.ravel()

    err = check_grad(func, grad, C.ravel())
    assert err < 1e-6


@pytest.mark.parametrize("operator", ['hardmax', 'softmax', 'sparsemax'])
def test_viterbi_hessian(operator):
    theta = make_data()
    Z = np.random.randn(*theta.shape)

    def func(X):
        X = X.reshape(theta.shape)
        _, grad, _, _ = dtw_grad(X, operator=operator)
        return np.sum(grad * Z)

    def grad(X):
        X = X.reshape(theta.shape)
        v, H = dtw_hessian_prod(X, Z, operator=operator)
        return H.ravel()

    # check_grad does not work with ndarray of dim > 2
    err = check_grad(func, grad, theta.ravel())
    assert err < 1e-6
