import torch.nn.functional as F
import torch
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1, device='cpu'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.RNN_type = rnn_type
        self.device = device

        self.embedding = nn.Embedding(input_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self.rnn = nn.__dict__[self.RNN_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=dropout_p,
            batch_first=True
        )

    def forward(self, encoder_input):
        encoder_input = self.embedding(encoder_input)
        encoder_input = self.dropout(encoder_input)
        output, hidden = self.rnn(encoder_input)

        # last layer hidden state, all hidden state
        return hidden, output


class DecoderCell(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1,
                 device='cpu', max_length=100):
        super(DecoderCell, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.RNN_type = rnn_type

        self.embedding = nn.Embedding(output_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self.rnn = nn.__dict__[self.RNN_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=dropout_p,
            batch_first=True
        )

        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, decoder_input, hidden):
        output = self.embedding(decoder_input)
        output = self.dropout(output)
        # output = F.relu(output)

        output, hidden = self.rnn(output, hidden)

        output = self.out(output[:, -1, :])
        output = self.softmax(output)
        return output, hidden


class AdditiveAttention(nn.Module):
    """Additive attention."""
    attention_weights: torch.Tensor

    def __init__(self, key_size, query_size, num_hiddens, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)

    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries + keys
        features = torch.tanh(features)
        scores = self.w_v(features)
        self.attention_weights = F.softmax(scores, dim=0)

        bmm = torch.bmm(self.attention_weights, values)
        return torch.sum(bmm, dim=0)


class AttnDecoderCell(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, rnn_type,
                 dropout_p=0., device='cpu', max_length=100):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens=hidden_size, key_size=hidden_size, query_size=hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(
            hidden_size * 2, hidden_size, num_layers,
            dropout=dropout_p,
            batch_first=True)
        self.dense = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, decoder_input, hidden_state, enc_outputs):
        enc_outputs = enc_outputs.unsqueeze(0).permute(1, 0, 2)

        embedded = self.embedding(decoder_input)
        embedded = self.dropout(embedded)

        query = torch.unsqueeze(hidden_state[-1], dim=1)
        context = self.attention(
            query, enc_outputs, enc_outputs)

        context = context.unsqueeze(0)

        x = torch.cat((context, embedded), dim=-1)

        outputs, hidden_state = self.rnn(x, hidden_state)
        x = torch.cat((context, hidden_state), dim=-1)
        outputs = F.log_softmax(self.dense(x[0]), dim=1)
        return outputs, hidden_state


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1,
                 attention=False, device='cpu', max_length=100):
        super(DecoderRNN, self).__init__()

        self.attention = attention
        if attention:
            self.decoder_cell = AttnDecoderCell(output_size, hidden_size, n_layers, rnn_type, dropout_p, device,
                                                max_length)
        else:
            self.decoder_cell = DecoderCell(output_size, hidden_size, n_layers, rnn_type, dropout_p, device)

    def forward(self, decoder_input, hidden, enc_outputs=None):
        assert enc_outputs is not None if self.attention else True
        # If attention is used, all encoder hidden states must be provided
        if self.attention:
            output, hidden = self.decoder_cell(decoder_input, hidden, enc_outputs)
        else:
            output, hidden = self.decoder_cell(decoder_input, hidden)

        return output, hidden
