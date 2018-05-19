"""Reference smoothed max/min 1D operations."""


from typing import Tuple

import numpy as np


class BaseOp:
    """
    Base class for smoothed max/min operation
    """

    @staticmethod
    def max(x: np.ndarray) -> Tuple[float, np.ndarray]:
        raise NotImplementedError

    @classmethod
    def min(cls, x: np.ndarray) -> Tuple[float, np.ndarray]:
        min_x, argmax_x = cls.max(-x)
        return - min_x, argmax_x

    @classmethod
    def argmax(cls, x: np.ndarray) -> np.ndarray:
        return cls.min(x)[1]

    @classmethod
    def argmin(cls, x: np.ndarray) -> np.ndarray:
        return cls.max(x)[1]

    @staticmethod
    def hessian_product(p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def min_jacobian(cls, p: np.ndarray) -> np.ndarray:
        return - cls.hessian_product(p)


class SoftMaxOp(BaseOp):
    """
    The soft(arg)max operations.
    """

    @staticmethod
    def max(x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Solves $max_{p \in \Delta^d} <x, p> - \sum_{i=1}^d p_i \log(p_i)$.

        :param x: np.ndarray, shape = (n,)
            Vector to project
        :return: Tuple[float, np.ndarray]
            max(x), argmax(x)
        """
        max_x = np.max(x)
        exp_x = np.exp(x - max_x)
        Z = np.sum(exp_x)
        return np.log(Z) + max_x, exp_x / Z

    @staticmethod
    def hessian_product(p: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute the product of max hessian with z using the gradient p of max.

        :param p: np.ndarray, shape = (p)
            Gradient of max(x)
        :param z: np.ndarray, shape = (p)
            Vector to compute the Jacobian product with
        :return: np.ndarray, shape = (p, p)
            Jacobian of argmax(x)
        """
        return p * z - p * np.sum(p * z)


class SparseMaxOp(BaseOp):
    """
    The sparsemax operations.
    """

    @classmethod
    def max(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Solves max_{p \in \Delta^d} <x, p> - \frac{1}{2} \sum_{i=1}^d p_i^2.

        :param x: np.ndarray, shape = (n,)
            Vector to project
        :return: Tuple[float, np.ndarray]
            max(x), argmax(x)
        """
        n_features = x.shape[0]
        u = np.sort(x)[::-1]
        z = np.ones_like(x)
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        proj_x = np.maximum(x - theta, 0.)
        min_x: float = np.sum(proj_x * (x - .5 * proj_x))
        return min_x, proj_x

    @staticmethod
    def hessian_product(p: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute the product of max hessian with z using the gradient p of max.

        :param p: np.ndarray, shape = (p)
            Gradient of max(x)
        :param z: np.ndarray, shape = (p)
            Vector to compute the Jacobian product with
        :return: np.ndarray, shape = (p, p)
            Jacobian of argmax(x)
        """
        s = p > 0
        return s * z - s * (np.sum(s, z) / np.sum(s))


class HardMaxOp(BaseOp):
    """
    The regular max operations.
    """

    @staticmethod
    def max(x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Solves max_{p \in \Delta^d} <x, p>

        :param x: np.ndarray, shape = (n,)
            Vector to project
        :return: Tuple[float, np.ndarray]
            max(x), argmax(x)
        """
        i = np.argmax(x)
        argmax_x = np.zeros_like(x)
        argmax_x[i] = 1
        max_x = x[i]
        return max_x, argmax_x

    @staticmethod
    def hessian_product(p: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute the product of max hessian with z using the gradient p of max.

        :param p: np.ndarray, shape = (p)
            Gradient of max(x)
        :param z: np.ndarray, shape = (p)
            Vector to compute the Jacobian product with
        :return: np.ndarray, shape = (p, p)
            Jacobian of argmax(x)
        """
        n_features = p.shape[0]
        return np.zeros(n_features)


operators = {'softmax': SoftMaxOp, 'sparsemax': SparseMaxOp,
             'hardmax': HardMaxOp}