"""Reference smoothed max/min 1D operations."""


from typing import Tuple

import numpy as np


class BaseOp:
    """Base class for smoothed max/min operation"""
    def max(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        raise NotImplementedError

    def min(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        min_x, argmax_x = self.max(-x)
        return - min_x, argmax_x

    def argmax(self, x: np.ndarray) -> np.ndarray:
        return self.min(x)[1]

    def argmax(self, x: np.ndarray) -> np.ndarray:
        return self.max(x)[1]

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def min_jacobian(self, p: np.ndarray) -> np.ndarray:
        return - self.jacobian(p)


class SoftMaxOp(BaseOp):
    """The soft(arg)max operations.

    Solves max_{p \in \Delta^d} <x, p> - \sum_{i=1}^d p_i \log(p_i)
    """
    def max(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        max_x = np.max(x)
        exp_x = np.exp(x - max_x)
        Z = np.sum(exp_x)
        return np.log(Z) + max_x, exp_x / Z

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        return np.diag(p) - np.outer(p, p)


class SparseMaxOp(BaseOp):
    """The sparsemax operations.

    Solves max_{p \in \Delta^d} <x, p> - \frac{1}{2} \sum_{i=1}^d p_i^2
    """
    def max(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        n_features = x.shape[0]
        u = np.sort(x)[::-1]
        z = np.ones_like(x)
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        proj_x = np.clip(x - theta, 0., None)
        min_x: float = np.sum(proj_x * (x - .5 * proj_x))
        return min_x, proj_x

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        s = p > 0
        return np.diag(s) - np.outer(s, s) / np.sum(s)


class HardMaxOp(BaseOp):
    """The regular max operations.

        Solves max_{p \in \Delta^d} <x, p>
    """
    def max(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        i = np.argmax(x)
        argmax_x = np.zeros_like(x)
        argmax_x[i] = 1
        max_x = x[i]
        return max_x, argmax_x

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        n_features = p.shape[0]
        return np.zeros((n_features, n_features))