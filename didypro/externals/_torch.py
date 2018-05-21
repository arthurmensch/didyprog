from functools import reduce

import numpy as np
import torch
from scipy.optimize import fmin_l_bfgs_b
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Optimizer

eps = np.finfo('double').eps

def pad_sequence(sequences, batch_first=False):
    r"""Pad a list of variable length Variables with zero

    ``pad_sequence`` stacks a list of Variables along a new dimension,
    and padds them to equal length. For example, if the input is list of
    sentences with size ``Lx*`` and if batch_first is False, and ``TxBx*``
    otherwise. The list of sentences should be sorted in the order of
    decreasing length.

    B is batch size. It's equal to the number of elements in ``sentences``.
    T is length longest sequence.
    L is length of the sequence.
    * is any number of trailing dimensions, including none.

    Example:
        >>> a = Variable(torch.ones(25, 300))
        >>> b = Variable(torch.ones(22, 300))
        >>> c = Variable(torch.ones(15, 300))
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Variable of size TxBx* or BxTx* where T is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Variables
            in sentences are same.

    Arguments:
        sentences (list[Variable]): list of variable length sentences.
        batch_first (bool, optional): output will be in BxTx* if True, or in
            TxBx* otherwise

    Returns:
        Variable of size ``T x B x * `` if batch_first is False
        Variable of size ``B x T x * `` otherwise
    """

    # assuming trailing dimensions and type of all the Variables
    # in sentences are same and fetching those from sentences[0]
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
        batch_dim = 0
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
        batch_dim = 1

    out_variable = Variable(sequences[0].data.new(*out_dims).zero_())
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
            raise ValueError(
                "lengths array has to be sorted in decreasing order")
        prev_l = length
        if batch_first:
            out_variable[i, :length] = variable
        else:
            out_variable[:length, i] = variable

    return out_variable


def pack_sequence(sequences):
    r"""Packs a list of variable length Variables

    ``sentences`` should be a list of Variables of size ``Lx*``, where L is
    the length of a sequence and * is any number of trailing dimensions,
    including zero. They should be sorted in the order of decreasing length.

    Example:
        >>> a = Variable(torch.Tensor([1,2,3]))
        >>> b = Variable(torch.Tensor([4,5]))
        >>> c = Variable(torch.Tensor([6]))
        >>> pack_sequence([a, b, c]])
        PackedSequence(data=
         1
         4
         6
         2
         5
         3
        [torch.FloatTensor of size 6]
        , batch_sizes=[3, 2, 1])


    Arguments:
        sentences (list[Variable]): A list of sentences of decreasing length.

    Returns:
        a :class:`PackedSequence` object
    """
    return pack_padded_sequence(pad_sequence(sequences),
                                [v.size(0) for v in sequences])


def cat_padded_sequence(sequence, lengths):
    output = []
    for i, length in enumerate(lengths):
        output.extend(sequence[:length, i][None, :])
    return torch.cat(output)


class LBFGSScipy(Optimizer):
    """Wrap L-BFGS algorithm, using scipy routines.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now CPU only

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """

    def __init__(self, params, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10,
                 ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size)
        super(LBFGSScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        self._n_iter = 0
        self._last_loss = None
        self._pinned_grad = None
        self._pinned_params = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            if p.is_cuda:
                view = view.cpu()
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            if p.is_cuda:
                view = view.cpu()
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            data = params[offset:offset + numel].view_as(p.data)
            if p.is_cuda:
                device = p.get_device()
                data = data.cuda(device)
            p.data = data
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns `the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        history_size = group['history_size']

        self._pinned_params = self._gather_flat_params()

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            self._pinned_params[:] = torch.from_numpy(flat_params)
            self._distribute_flat_params(self._pinned_params)
            loss = closure()
            self._last_loss = loss
            loss = loss.data[0]
            flat_grad = self._gather_flat_grad()
            if self._pinned_grad is None:
                self._pinned_grad = flat_grad
            else:
                self._pinned_grad[:] = flat_grad
            return loss, self._pinned_grad.numpy().astype(np.float64)

        def callback(flat_params):
            self._n_iter += 1
            print('Iter %i Loss %.5f' % (self._n_iter, self._last_loss.data[0]))

        fmin_l_bfgs_b(wrapped_closure, self._pinned_params.numpy().astype(np.float64), maxiter=max_iter,
                      maxfun=max_eval,
                      factr=tolerance_change / eps, pgtol=tolerance_grad, epsilon=0,
                      # disp=100,
                      m=history_size,
                      callback=callback)