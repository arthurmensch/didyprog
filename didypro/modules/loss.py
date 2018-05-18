import torch
from torch import nn
from torch.nn import NLLLoss
from torch.nn.utils.rnn import pack_padded_sequence


class BinaryMSELoss(nn.Module):
    def forward(self, pred, target, lengths):
        with torch.cuda.device_of(lengths):
            lengths = lengths.tolist()
        pred = pack_padded_sequence(pred, lengths, batch_first=True)[0]
        target = pack_padded_sequence(target, lengths, batch_first=True)[0]
        return (torch.sum(pred ** 2) + pred.shape[0] - 2 * torch.sum(
            torch.gather(pred, dim=1, index=target[:, None]))) / pred.shape[0]


class BinaryL1Loss(nn.Module):
    def forward(self, pred, target, lengths):
        with torch.cuda.device_of(lengths):
            lengths = lengths.tolist()
        pred = pack_padded_sequence(pred, lengths, batch_first=True)[0]
        target = pack_padded_sequence(target, lengths, batch_first=True)[0]
        return (torch.sum(pred) + pred.shape[0] - 2 * torch.sum(
            torch.gather(pred, dim=1, index=target[:, None]))) / pred.shape[0]


class OurNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.nllloss = NLLLoss()

    def forward(self, pred, target, lengths):
        with torch.cuda.device_of(lengths):
            lengths = lengths.tolist()
        pred = pack_padded_sequence(pred, lengths, batch_first=True)[0]
        target = pack_padded_sequence(target, lengths, batch_first=True)[0]
        return self.nllloss(torch.log(pred), target)
