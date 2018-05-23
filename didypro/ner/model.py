import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from didypro.ner.viterbi import PackedViterbi
from didypro.ner.potential import LinearPotential


class GatedCNNProcessor(nn.Module):
    def __init__(self, in_channels, out_channels=100,
                 n_layers=1, dropout=0.):
        super(GatedCNNProcessor, self).__init__()

        cnns = []
        gating_cnns = []

        cnns.append(nn.Conv1d(in_channels, out_channels,
                              kernel_size=3))
        gating_cnns.append(nn.Conv1d(in_channels, out_channels,
                                     kernel_size=3))
        for i in range(n_layers - 1):
            cnns.append(nn.Conv1d(out_channels, out_channels,
                                  kernel_size=3, ))
            gating_cnns.append(nn.Conv1d(out_channels, out_channels,
                                         kernel_size=3, ))
        self.gating_cnns = nn.ModuleList(gating_cnns)
        self.cnns = nn.ModuleList(cnns)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for cnn, gating_cnn in zip(self.cnns, self.gating_cnns):
            cnn.reset_parameters()
            gating_cnn.reset_parameters()

    def forward(self, inputs, lengths, sorted=True):
        # batch, seq_len, features
        prev = inputs.transpose(1, 2)
        for i, (cnn, gating_cnn) in enumerate(
                zip(self.cnns, self.gating_cnns)):
            current = F.pad(prev, (1, 1))
            current = self.sigmoid(gating_cnn(current)) * cnn(current)
            current = self.dropout(current)
            prev = current
        return current.transpose(1, 2)


class ConvPoolProcessor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPoolProcessor, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, inputs, lengths, sorted=True):
        # batch, seq_len, features
        prev = inputs.transpose(1, 2)
        # batch, features, seq_len
        inputs = F.pad(prev, (1, 1))
        conv = self.conv(inputs)
        pooled, _ = torch.max(conv, dim=2)
        pooled = torch.tanh(pooled)
        return pooled


class TanhUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super(TanhUnit, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight,
                               nn.init.calculate_gain('tanh'))
        self.linear.bias.data.zero_()

    def forward(self, input):
        return self.tanh(self.linear(input))


class LSTMProcessor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pad=False,
                 return_type='last'):
        super(LSTMProcessor, self).__init__()

        self.pad = pad

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True)
        self.return_type = return_type

        if self.return_type == 'all':
            self.merge = TanhUnit(2 * hidden_dim, hidden_dim)
        elif not self.return_type == 'last':
            raise NotImplementedError

    def _init_lstm_hidden(self, batch_size=1):
        new = next(self.parameters()).data.new
        hidden_size = self.lstm.hidden_size
        return (new(2, batch_size, hidden_size).zero_(),
                new(2, batch_size, hidden_size).zero_())

    def reset_parameters(self):
        self.lstm.reset_parameters()
        if hasattr(self, 'merge'):
            self.merge.reset_parameters()

    def forward(self, inputs, lengths, sorted=False):
        n_batch, seq_len, feature_size = inputs.shape

        if not sorted:
            _, indices = torch.sort(lengths, descending=True)
            _, rev_indices = torch.sort(indices, descending=False)
            inputs = inputs[indices]
            lengths = lengths[indices]
            zero_cut = torch.nonzero(lengths == 0)
            if len(zero_cut) > 0:
                zero_cut = zero_cut[0, 0].item()
                inputs = inputs[:zero_cut]
                lengths = lengths[:zero_cut]
                pad_size = n_batch - zero_cut
            else:
                pad_size = 0

        embeds = pack_padded_sequence(inputs, lengths, batch_first=True)

        batch_size = inputs.shape[0]
        lstm_hidden = self._init_lstm_hidden(batch_size)
        lstm_out, (lstm_hidden, _) = self.lstm(embeds, lstm_hidden)
        if self.return_type == 'last':
            bidir_out = torch.cat([lstm_hidden[0], lstm_hidden[1]], dim=1)
            if not sorted:
                if pad_size > 0:
                    bidir_out = F.pad(bidir_out, (0, 0, 0, pad_size))
                bidir_out = bidir_out[rev_indices]
            return bidir_out
        else:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            lstm_out = self.merge(lstm_out)
            if not sorted:
                if pad_size > 0:
                    lstm_out = F.pad(lstm_out, (0, 0, 0, 0, 0,
                                                pad_size))
                lstm_out = lstm_out[rev_indices]
            return lstm_out


class CharWordEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size,
                 letter_proc='lstm',
                 letter_embedding_dim=None, letter_size=None,
                 padding_idx=1,
                 letter_hidden_dim=None):
        super(CharWordEmbedding, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim,
                                            padding_idx=padding_idx)
        self.embedding_dim = embedding_dim
        if letter_proc in ['lstm', 'conv']:
            self.letter_embeddings = nn.Embedding(letter_size,
                                                  letter_embedding_dim,
                                                  padding_idx=padding_idx)
            if letter_proc == 'lstm':
                self.letter_processor = LSTMProcessor(letter_embedding_dim,
                                                      letter_hidden_dim // 2)
                self.embedding_dim += letter_hidden_dim
            else:
                self.letter_processor = ConvPoolProcessor(letter_embedding_dim,
                                                          letter_hidden_dim)
                self.embedding_dim += letter_hidden_dim
        elif letter_proc is not None:
            raise NotImplementedError

    def reset_parameters(self):
        self.word_embeddings.reset_parameters()
        if hasattr(self, 'letter_embeddings'):
            self.letter_embeddings.reset_parameters()
        if hasattr(self, 'letter_proc'):
            self.letter_processor.reset_parameters()

    def forward(self, sentences, lengths, letters, letters_lengths):
        batch_size, seq_len = sentences.shape
        embeds = self.word_embeddings(sentences.view(-1)).view(batch_size,
                                                               seq_len, -1)
        if hasattr(self, 'letter_embeddings'):
            batch_size, seq_len, letter_len = letters.shape
            letters = letters.view((-1, letter_len))
            letters_lengths = letters_lengths.view(-1)
            letter_embeds = self.letter_embeddings(letters.view(-1)).view(
                batch_size, seq_len, letter_len, -1)
            letter_embedding_dim = letter_embeds.shape[3]
            letter_embeds = letter_embeds.view(
                (-1, letter_len, letter_embedding_dim))

            letters_proc = self.letter_processor(letter_embeds,
                                                 letters_lengths)
            letters_proc = letters_proc.view((batch_size, seq_len, -1))
            embeds = torch.cat([embeds, letters_proc], dim=2)
        return embeds


class Tagger(nn.Module):
    """
        Tagger based on a BiLSTM layer + optional l2/entropy CRF smoothing
    """

    def __init__(self, embedding_dim, vocab_size, tagset_size,
                 hidden_dim=None,
                 padding_idx=1,
                 letter_proc='lstm',
                 proc='lstm',
                 dropout=True,
                 alpha=1, operator='softmax',
                 eos_idx=None, init_idx=None,
                 letter_embedding_dim=None, letter_size=None,
                 letter_hidden_dim=None):
        super(Tagger, self).__init__()

        self.embedder = CharWordEmbedding(embedding_dim, vocab_size,
                                          padding_idx=padding_idx,
                                          letter_proc=letter_proc,
                                          letter_embedding_dim=
                                          letter_embedding_dim,
                                          letter_size=letter_size,
                                          letter_hidden_dim=letter_hidden_dim)
        if proc == 'lstm':
            self.processor = LSTMProcessor(self.embedder.embedding_dim,
                                           hidden_dim,
                                           return_type='all')
        elif proc == 'gcnn':
            self.processor = GatedCNNProcessor(self.embedder.embedding_dim,
                                               hidden_dim, n_layers=3,
                                               dropout=dropout,
                                               )
        else:
            raise NotImplementedError

        if dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.alpha = alpha

        self.linear_potential = LinearPotential(hidden_dim, tagset_size,
                                                eos_idx=eos_idx,
                                                init_idx=init_idx)
        self.viterbi = PackedViterbi(operator=operator)

    def reset_parameters(self):
        self.embedder.reset_parameters()
        self.processor.reset_parameters()
        self.linear_potential.reset_parameters()

    def _get_potentials(self, sentences, lengths, letters, letters_lengths,
                        sorted=False):
        if not sorted:
            _, indices = torch.sort(lengths, descending=True)
            _, rev_indices = torch.sort(indices, descending=False)
            sentences = sentences[indices]
            lengths = lengths[indices]
            letters = letters[indices]
            letters_lengths = letters[indices]
        else:
            rev_indices = None

        embeds = self.embedder(sentences, lengths, letters, letters_lengths)
        embeds = self.dropout(embeds)
        proc = self.processor(embeds, lengths, sorted=True)
        potential = self.linear_potential(proc)
        potential = pack_padded_sequence(potential, lengths, batch_first=True)
        return potential, rev_indices

    def forward(self, sentences, lengths, letters=None, letters_lengths=None,
                sorted=False):
        potentials, rev_indices = self._get_potentials(sentences, lengths,
                                                       letters,
                                                       letters_lengths,
                                                       sorted=sorted)

        scores = self.viterbi.decode(potentials)
        scores, _ = pad_packed_sequence(scores, batch_first=True)
        if not sorted:
            scores = scores[rev_indices]
        scores = scores.sum(dim=3)
        return scores

    def partition_potentials(self, sentences, lengths, letters=None,
                             letters_lengths=None, sorted=False):
        potentials, rev_indices = self._get_potentials(sentences, lengths,
                                                       letters,
                                                       letters_lengths,
                                                       sorted=sorted)
        partition = self.viterbi(potentials)

        potentials, _ = pad_packed_sequence(potentials, batch_first=True)
        if not sorted:
            potentials = potentials[rev_indices]

        return partition, potentials