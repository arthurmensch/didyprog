from typing import Tuple

from didypro.reference.np.local import operators

import numpy as np


def viterbi_value(theta: np.ndarray, operator: str = 'hardmax') \
        -> float:
    """
    Viterbi operator.

    :param theta: np.ndarray, shape = (T, S, S),
        Holds the potentials of the linear chain CRF
    :param operator: str in {'hardmax', 'softmax', 'sparsemax'},
        Smoothed max-operator
    :return: float,
        DTW value $Vit(\theta)$
    """
    return viterbi_grad(theta, operator)[0]


def viterbi_grad(theta: np.ndarray,
                 operator: str = 'hardmax') \
        -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Value and gradient of the Viterbi operator.

    Algorithm 3 in the paper.

    :param theta: np.ndarray, shape = (T, S, S),
        Holds the potentials of the linear chain CRF
    :param operator: str in {'hardmax', 'softmax', 'sparsemax'},
        Smoothed max-operator
    :return: Tuple[float, np.ndarray, np.ndarray, np.ndarray],
        v: float,
            Viterbi value $Vit(\theta)$
        grad: np.ndarray, shape = (T, S, S),
            Viterbi gradient, $\nabla Vit(\theta)$
        Q: np.ndarray,
            Intermediary computations
        U: np.ndarray,
            Intermediary computations
    """
    operator = operators[operator]
    T, S, _ = theta.shape
    V = np.zeros((T + 1, S))
    Q = np.zeros((T + 2, S, S))
    U = np.zeros((T + 2, S))
    E = np.zeros((T + 1, S, S))
    # V[0, 1:] = -1e10
    for t in range(1, T + 1):
        for i in range(S):
            # t - 1 corresponds to padding vector theta
            V[t, i], Q[t, i] = operator.max(theta[t - 1, i] + V[t - 1])
    v, Q[T + 1, 0] = operator.max(V[T])
    U[T + 1, 0] = 1
    for t in reversed(range(0, T + 1)):
        E[t] = Q[t + 1] * U[t + 1][:, np.newaxis]
        U[t] = np.sum(E[t], axis=0)
    return v, E[:-1], Q, U


def viterbi_hessian_prod(theta: np.ndarray, Z: np.ndarray,
                         operator: str = 'hardmax') -> Tuple[float, np.ndarray]:
    """
    Dir. derivative and Hessian-vector product of the Viterbi operator.

    Algorithm 4 in the paper.

    :param theta: np.ndarray, shape = (T, S, S)
        Holds the potentials of the linear chain CRF
    :param Z: np.ndarray, shape = (T, S, S)
        Direction in which to compute the Hessian-vector product
    :param operator: str in {'hardmax', 'softmax', 'sparsemax'},
        Smoothed max-operator
    :return: Tuple[float, np.ndarray],
        vdot: directional derivative
            $<\nabla Vit(\theta), Z>$
        hessian_prod: np.ndarray, (T, S, S),
            Hessian product $\nabla^2 Vit(\theta) Z$
    """
    _, _, Q, U = viterbi_grad(theta, operator)
    operator = operators[operator]

    T, S, _ = Z.shape
    Vdot = np.zeros((T + 1, S))
    Qdot = np.zeros((T + 2, S, S))
    Udot = np.zeros((T + 2, S))
    Edot = np.zeros((T + 1, S, S))

    for t in range(1, T + 1):
        for i in range(S):
            M = Z[t - 1, i] + Vdot[t - 1]
            Vdot[t, i] = np.sum(Q[t, i] * M)
            H = operator.jacobian(Q[t, i])
            Qdot[t, i] = H.dot(M)
    vdot: float = np.sum(Q[T + 1, 0] * Vdot[T])
    H = operator.jacobian(Q[T + 1, 0])
    Qdot[T + 1, 0] = H.dot(Vdot[T])
    for t in reversed(range(0, T + 1)):
        Edot[t] = (Q[t + 1] * Udot[t + 1][:, None]
                   + Qdot[t + 1] * U[t + 1][:, None])
        Udot[t] = np.sum(Edot[t], axis=0)
    return vdot, Edot[:-1]
