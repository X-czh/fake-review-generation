import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

from utils import SOS_token


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor, input_lengths, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder(encoder_hidden, max_length)
        return decoder_outputs, decoder_hidden


class EncoderBiRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(EncoderBiRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, 
            num_layers=n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths)
        packed_output, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(packed_output) # unpack (back to padded)
        outputs = torch.cat([
            outputs[:, :, :self.hidden_size], 
            outputs[:, :, self.hidden_size:]
            ], 2)  # Cat bidirectional outputs
        return outputs, hidden


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, noise_size, n_layers, dropout, use_cuda):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_size = noise_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size + noise_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward_step(self, inputs, hidden, noise):
        embedded = self.embedding(inputs)
        output, hidden = self.gru(embedded, hidden)
        output = output.squeeze(0)  # squeeze the time dimension
        
        output = torch.cat((noise, output), dim=1)

        output = self.log_softmax(self.fc(output))
        return output, hidden

    def forward(self, context_vector, max_length):

        # Prepare tensor for decoder on time_step_0
        batch_size = context_vector.size(1)
        decoder_input = torch.LongTensor([[SOS_token] * batch_size])

        # Pass the context vector
        decoder_hidden = context_vector

        decoder_outputs = torch.zeros(
            max_length,
            batch_size,
            self.output_size
        ) # (time_steps, batch_size, vocab_size)

        # Sample noise
        noise = torch.Tensor(np.random.normal(0, 1, (batch_size, self.noise_size)))

        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()
            noise = noise.cuda()

        # Unfold the decoder RNN on the time dimension
        for t in range(max_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, noise)
            decoder_outputs[t] = decoder_output
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.transpose(0, 1)

        return decoder_outputs, decoder_hidden
