import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.stats import multivariate_normal

from didypro.reference.local import HardMaxOp, SoftMaxOp, SparseMaxOp
from didypro.reference.viterbi import viterbi_grad, viterbi_value, \
    viterbi_hessian_product


def sample(transition_matrix,
           means, covs,
           start_state, n_samples,
           random_state):
    n_states, n_features, _ = covs.shape
    states = np.zeros(n_samples, dtype='int')
    emissions = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        if i == 0:
            prev_state = start_state
        else:
            prev_state = states[i - 1]
        state = random_state.choice(n_states,
                                    p=transition_matrix[:, prev_state])
        emissions[i] = random_state.multivariate_normal(means[state],
                                                        covs[state])
        states[i] = state
    return emissions, states


def make_data(T=20):
    """Sample data from a HMM model and compute associated CRF potentials."""
    random_state = np.random.RandomState(0)

    transition_matrix = np.array([[0.5, 0.1, 0.1],
                                  [0.3, 0.5, 0.1],
                                  [0.2, 0.4, 0.8]
                                  ])
    means = np.array([[0, 0],
                      [10, 0],
                      [5, -5]
                      ])
    covs = np.array([[[1, 0],
                      [0, 1]],
                     [[.2, 0],
                      [0, .3]],
                     [[2, 0],
                      [0, 1]]
                     ])
    start_state = 0

    emissions, states = sample(transition_matrix, means, covs, start_state,
                               n_samples=T, random_state=random_state)
    emission_log_likelihood = []
    for mean, cov in zip(means, covs):
        rv = multivariate_normal(mean, cov)
        emission_log_likelihood.append(rv.logpdf(emissions)[:, np.newaxis])
    emission_log_likelihood = np.concatenate(emission_log_likelihood, axis=1)
    log_transition_matrix = np.log(transition_matrix)

    # CRF potential from HMM model
    theta = emission_log_likelihood[:, :, np.newaxis] \
            + log_transition_matrix[np.newaxis, :, :]

    return states, emissions, theta


@pytest.mark.parametrize("operator", [HardMaxOp, SoftMaxOp, SparseMaxOp])
def test_viterbi(operator):
    states, emissions, theta = make_data()
    viterbi, grad, _, _ = viterbi_grad(theta, operator=operator)
    decoded = np.argmax(grad.sum(axis=2), axis=1)
    assert np.all(decoded == states)


@pytest.mark.parametrize("operator", [HardMaxOp, SoftMaxOp, SparseMaxOp])
def test_dtw_grad(operator):
    states, emissions, theta = make_data()

    def func(X):
        X = X.reshape(theta.shape)
        return viterbi_value(X, operator=operator)

    def grad(X):
        X = X.reshape(theta.shape)
        _, grad, _, _ = viterbi_grad(X, operator=operator)
        return grad.ravel()

    # check_grad does not work with ndarray of dim > 2
    check_grad(func, grad, theta.ravel())


@pytest.mark.parametrize("operator", [HardMaxOp, SoftMaxOp, SparseMaxOp])
def test_dtw_hessian(operator):
    states, emissions, theta = make_data()

    S = np.random.randn(*theta.shape)

    def func(X):
        X = X.reshape(theta.shape)
        _, grad, _, _ = viterbi_grad(X, operator=operator)
        return np.sum(grad * S)

    def grad(X):
        X = X.reshape(theta.shape)
        _, grad, U, Q = viterbi_grad(X, operator=operator)
        viterbi_hessian_product(U, Q, X, operator=operator)
        return grad.ravel()

    # check_grad does not work with ndarray of dim > 2
    check_grad(func, grad, theta.ravel())
