import numpy as np
import pytest
from scipy.optimize import check_grad
from sklearn.metrics.pairwise import pairwise_distances

from didypro.reference.dtw import dtw_value, dtw_grad, dtw_hessian_prod
from didypro.reference.local import SoftMaxOp, SparseMaxOp, HardMaxOp


def make_data():
    rng = np.random.RandomState(0)
    m, n = 2, 2
    X = rng.randn(m, 3)
    Y = rng.randn(n, 3)
    return pairwise_distances(X, Y) / 10


def test_dtw():
    C = make_data()
    dtw = dtw_value(C, operator=HardMaxOp)

    _, grad, _, _ = dtw_grad(C, operator=HardMaxOp)
    _, soft_grad, _, _ = dtw_grad(C, operator=SoftMaxOp)
    _, sparse_grad, _, _ = dtw_grad(C, operator=SparseMaxOp)

    assert(dtw == np.sum(grad * C))
    assert(dtw < np.sum(soft_grad * C))
    assert(dtw < np.sum(sparse_grad * C))


@pytest.mark.parametrize("operator", [HardMaxOp, SoftMaxOp, SparseMaxOp])
def test_dtw_grad(operator):
    C = make_data()

    def func(X):
        return dtw_value(X, operator=operator)

    def grad(X):
        _, grad, _, _ = dtw_grad(X, operator=operator)
        return grad

    check_grad(func, grad, C)



@pytest.mark.parametrize("operator", [HardMaxOp, SoftMaxOp, SparseMaxOp])
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
    check_grad(func, grad, theta.ravel())
