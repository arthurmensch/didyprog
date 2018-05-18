import numpy as np

from didypro.reference.numpy.local import HardMaxOp, SparseMaxOp, SoftMaxOp


def make_data():
    rng = np.random.RandomState(0)
    return rng.randint(-10, 10, size=10)


def test_hardmax():
    x = make_data()
    op = HardMaxOp()
    max_x, argmax_x = op.max(x)
    assert np.all(x <= max_x)
    assert np.sum(argmax_x * x) == max_x


def test_sparsemax():
    x = make_data()
    op = SparseMaxOp()
    max_x, argmax_x = op.max(x)
    assert np.all(argmax_x >= 0.)
    assert np.sum(argmax_x) == 1.


def test_softmax():
    x = make_data()
    op = SoftMaxOp()
    max_x, argmax_x = op.max(x)
    assert np.all(argmax_x >= 0.)
    assert np.sum(argmax_x) == 1.