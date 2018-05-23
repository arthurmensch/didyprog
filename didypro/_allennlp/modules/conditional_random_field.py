"""
Conditional random field
"""
import torch
from didypro._allennlp.modules.viterbi import viterbi, viterbi_decode
from torch.autograd import Variable
from typing import List, Tuple


class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    Parameters
    ----------
    num_tags : int, required
        The number of tags.
    constraints : List[Tuple[int, int]], optional (default: None)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to ``viterbi_tags()`` but do not affect ``forward()``.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : bool, optional (default: True)
        Whether to include the start and end transition parameters.
    """

    def __init__(self,
                 num_tags: int,
                 constraints: List[Tuple[int, int]] = None,
                 include_start_end_transitions: bool = True) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1) and end of sequence tags (num_tags + 2)
        if constraints is None:
            # All transitions are valid.
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(
                1.)
        else:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(
                0.)
            for i, j in constraints:
                constraint_mask[i, j] = 1.

        self._constraint_mask = torch.nn.Parameter(constraint_mask,
                                                   requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))


        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal(self.start_transitions)
            torch.nn.init.normal(self.end_transitions)

    def _make_potentials(self, logits: torch.Tensor,
                         mask: torch.LongTensor,
                         constrained: bool = False) -> torch.Tensor:
        sequence_length, batch_size, num_tags = logits.size()

        if constrained:
            start_tag = num_tags
            end_tag = num_tags + 1
            # Apply transition constraints
            transitions = (
                    self.transitions * self._constraint_mask[:num_tags,
                                       :num_tags]
                    + -10000.0 * (1 - self._constraint_mask[:num_tags,
                                      :num_tags])
            )

            if self.include_start_end_transitions:
                start_transitions = (
                        self.start_transitions
                        * self._constraint_mask[start_tag, :num_tags] +
                        -10000.0 *
                        (1 - self._constraint_mask[start_tag, :num_tags])
                )
                end_transitions = (
                        self.end_transitions
                        * self._constraint_mask[:num_tags, end_tag] +
                        -10000.0 * (1 - self._constraint_mask[
                                        :num_tags, end_tag])
                )
            else:
                start_transitions = -10000.0 * (
                        1 - self._constraint_mask[start_tag, :num_tags])
                end_transitions = -10000.0 * (
                        1 - self._constraint_mask[:num_tags, end_tag])
        else:
            transitions = self.transitions
            if self.include_start_end_transitions:
                start_transitions = self.start_transitions
                end_transitions = self.end_transitions

        # Transpose batch size and sequence dimensions

        potentials = (logits[:, :, :, None]
                      + transitions.transpose(0, 1)[None, None, :, :])
        potentials[0, :, :, 0] = logits[0]
        potentials[0, :, :, 1:] = -10000.0

        if self.include_start_end_transitions:
            potentials[0, :, :, 0] = (potentials[0, :, :, 0]
                                      + start_transitions[None, :])

            end_indices = torch.sum(mask.data, dim=0) - 1
            for i, index in enumerate(end_indices):
                potentials[index, i] = (potentials[index, i]
                                        + end_transitions[None, :, None])
        return potentials

    def _input_likelihood(self, logits: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood,
        which is the sum of the likelihoods across all possible state sequences.
        """
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        potentials = self._make_potentials(logits, mask, constrained=False)

        mask = mask.float()
        return viterbi(potentials, mask)

    def _joint_likelihood(self,
                          logits: torch.Tensor,
                          tags: torch.Tensor,
                          mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, num_tags = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        potentials = self._make_potentials(logits, mask, constrained=False)

        tags = tags.transpose(0, 1).data
        mask = mask.data.byte()
        path = torch.zeros(potentials.shape).byte()

        for b in range(batch_size):
            path[0, b, tags[0, b], 0] = 1
            for t in range(1, sequence_length):
                path[t, b, tags[t, b], tags[t - 1, b]] = mask[t, b]

        # for t in range(sequence_length):
        #     if t > 0:
        #         last_tags = tags[t - 1].tolist()
        #     else:
        #         last_tags = [0] * batch_size
        #     indices = [[t] * batch_size, list(range(batch_size)),
        #                tags[t].tolist(), last_tags]
        #     potential_selector[indices] = mask[t]
        return torch.sum(torch.masked_select(potentials, Variable(path)))

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        """
        Computes the log likelihood.
        """
        # pylint: disable=arguments-differ
        if mask is None:
            mask = torch.autograd.Variable(torch.ones(*tags.size()).long())

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self, logits: Variable, mask: Variable) -> List[
        List[int]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        """
        _, max_seq_length, num_tags = logits.size()
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        lengths = torch.sum(mask, dim=0).data.tolist()

        potentials = self._make_potentials(logits, mask, constrained=True)

        mask = mask.float()
        tags = viterbi_decode(potentials, mask)
        tags = torch.sum(tags, dim=3)
        _, tags = torch.max(tags, dim=2)
        tags = tags.transpose(0, 1).data
        tags = tags.tolist()
        tags = [tag[:length] for tag, length in zip(tags, lengths)]

        return tags
