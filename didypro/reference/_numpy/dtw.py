import numpy as np

from didypro.reference._numpy.local import operators
from typing import Tuple


def dtw_value(theta: np.ndarray, operator: str = 'hardmax') -> float:
    """
    DTW operator.

    :param theta: _numpy.ndarray, shape = (m, n),
        Distance matrix for DTW
    :param operator: str in {'hardmax', 'softmax', 'sparsemax'},
        Smoothed max-operator
    :return: float,
        DTW value, $DTW(\theta)$
    """
    return dtw_grad(theta, operator)[0]


def dtw_grad(theta: np.ndarray, operator: str = 'hardmax') \
        -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Value and gradient of the DTW operator.

    Algorithm 5 in the paper.

    :param theta: _numpy.ndarray, shape = (m, n),
        Distance matrix for DTW
    :param operator: str in {'hardmax', 'softmax', 'sparsemax'},
        Smoothed max-operator
    :return: Tuple[float, _numpy.ndarray],
        v: float,
            DTW value, $DTW(\theta)$
        grad: _numpy.ndarray, shape = (m, n),
            DTW gradient, $\nabla DTW(\theta)$
        Q: _numpy.ndarray
            Intermediary computations
        E: _numpy.ndarray,
            Intermediary computations
    """
    operator = operators[operator]

    m, n = theta.shape

    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            v, Q[i, j] = operator.min(np.array([V[i, j - 1],
                                                V[i - 1, j - 1],
                                                V[i - 1, j]]))
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in reversed(range(1, m + 1)):
        for j in reversed(range(1, n + 1)):
            E[i, j] = Q[i, j + 1, 0] * E[i, j + 1] + \
                      Q[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                      Q[i + 1, j, 2] * E[i + 1, j]

    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


def dtw_hessian_prod(theta, Z, operator: str = 'hardmax')\
        -> Tuple[float, np.ndarray]:
    """
    Dir. derivative and Hessian-vector product of the DTW operator.

    Algorithm 6 in the paper.

    :param theta: _numpy.ndarray, shape = (m, n)
        Distance matrix for DTW
    :param Z: _numpy.ndarray, shape = (m, n)
        Direction in which to compute the Hessian-vector product
    :param operator: str in {'hardmax', 'softmax', 'sparsemax'},
        Smoothed max-operator
    :return: Tuple[float, _numpy.ndarray],
        vdot: float,
            directional derivative $<\nabla DTW(\theta), Z>$
        hessian_prod: _numpy.ndarray, shape = (m, n),
            directional derivative $\nabla^2 DTW(\theta) Z$
    """
    _, _, Q, E = dtw_grad(theta, operator)
    operator = operators[operator]

    m, n = Z.shape

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            V_dot[i, j] = Z[i - 1, j - 1] + \
                          Q[i, j, 0] * V_dot[i, j - 1] + \
                          Q[i, j, 1] * V_dot[i - 1, j - 1] + \
                          Q[i, j, 2] * V_dot[i - 1, j]

            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = operator.min_hessian_product(Q[i, j], v)
    E_dot = np.zeros((m + 2, n + 2))

    for j in reversed(range(1, n + 1)):
        for i in reversed(range(1, m + 1)):
            E_dot[i, j] = Q_dot[i, j + 1, 0] * E[i, j + 1] + \
                          Q[i, j + 1, 0] * E_dot[i, j + 1] + \
                          Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                          Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                          Q_dot[i + 1, j, 2] * E[i + 1, j] + \
                          Q[i + 1, j, 2] * E_dot[i + 1, j]

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1]
