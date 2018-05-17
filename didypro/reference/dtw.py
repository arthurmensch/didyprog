import numpy as np

from didypro.reference.local import HardMaxOp


def dtw_value(C, operator=HardMaxOp):
    m, n = C.shape

    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # C is indexed starting from 0.
            v, _ = operator.min(np.array([V[i - 1, j],
                                          V[i - 1, j - 1],
                                          V[i, j - 1]]))
            V[i, j] = C[i - 1, j - 1] + v

    return V[m, n]


def dtw_grad(C, operator=HardMaxOp):
    m, n = C.shape

    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # C is indexed starting from 0.
            v, Q[i, j] = operator.min(np.array([V[i, j - 1],
                                                V[i - 1, j - 1],
                                                V[i - 1, j]]))
            V[i, j] = C[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    # Q[m + 1, n + 1, 1] = 1
    Q[m + 1, n + 1] = 1

    for i in reversed(range(1, m + 1)):
        for j in reversed(range(1, n + 1)):
            E[i, j] = Q[i, j + 1, 0] * E[i, j + 1] + \
                      Q[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                      Q[i + 1, j, 2] * E[i + 1, j]

    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


def dtw_hessian_prod(M, Q, E, operator=HardMaxOp):
    m, n = M.shape

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # C is indexed starting from 0.
            V_dot[i, j] = M[i - 1, j - 1] + \
                          Q[i, j, 0] * V_dot[i, j - 1] + \
                          Q[i, j, 1] * V_dot[i - 1, j - 1] + \
                          Q[i, j, 2] * V_dot[i - 1, j]

            H = operator.min_jacobian(Q[i, j])
            v = [V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]]
            Q_dot[i, j] = np.dot(H, v)
    E_dot = np.zeros((m + 2, n + 2))

    for j in reversed(range(1, n + 1)):
        for i in reversed(range(1, m + 1)):
            E_dot[i, j] = Q_dot[i, j + 1, 0] * E[i, j + 1] + \
                          Q[i, j + 1, 0] * E_dot[i, j + 1] + \
                          Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                          Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                          Q_dot[i + 1, j, 2] * E[i + 1, j] + \
                          Q[i + 1, j, 2] * E_dot[i + 1, j]

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1], Q_dot
