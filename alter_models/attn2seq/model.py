import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SOS_token


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)

    def forward(self, input_seqs):
        return self.embedding(input_seqs)


class Attn(nn.Module):
    """ Implements modified LuongAttn mechanism on the output features from the encoder and decoder """

    def __init__(self, method, hidden_size, use_cuda):
        super().__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, output, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Create tensor to store attention energies
        attn_energies = torch.zeros(batch_size, seq_len) # B x S
        if self.use_cuda:
            attn_energies = attn_energies.cuda()

        # Calculate energies for encoder outputs
        attn_energies = self.score(output, encoder_outputs)

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies, 2)

    def score(self, output, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        output = output.transpose(0, 1) # 1 x B x N -> B x 1 x N
        encoder_outputs = encoder_outputs.transpose(0, 1) # S x B x N -> B x S x N

        if self.method == 'dot':
            encoder_outputs = encoder_outputs.transpose(2, 1) # B x N x S
            energy = torch.bmm(output, encoder_outputs) # B x 1 x S
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_outputs) # B x S x N
            energy = energy.transpose(2, 1) # B x S x N -> B x N x S
            energy = torch.bmm(output, energy) # B x 1 x S
            return energy

        elif self.method == 'concat':
            v = self.v.repeat(batch_size, 1, 1) # B x 1 x N
            output = output.repeat(1, seq_len, 1) # B x S x N
            energy = self.attn(torch.cat((output, encoder_outputs), 2)) # B x S x N
            energy = energy.transpose(2, 1) # B x S x N -> B x N x S
            energy = torch.bmm(v, F.tanh(energy)) # B x 1 x S
            return energy


def forward_step(self, inputs, hidden):
    embedded = self.embedding(inputs)
    output, hidden = self.gru(embedded, hidden)
    output = output.squeeze(0)  # squeeze the time dimension
    output = self.log_softmax(self.fc(output))
    return output, hidden


class AttnDecoderRNN(nn.Module):
    """ Implements modified LuongAttnDecoderRNN on the output features from the encoder and decoder """

    def __init__(self, attn_model, hidden_size, output_size, n_layers, dropout, use_cuda):
        super().__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, use_cuda)

    def forward_step(self, inputs, hidden, encoder_outputs):

        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(inputs)
        embedded = self.embedding_dropout(embedded)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, hidden)

        # Calculate attention from current RNN output and all encoder output states;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) # B x 1 x S
        encoder_outputs = encoder_outputs.transpose(0, 1) # B x S x N
        context = torch.bmm(attn_weights, encoder_outputs).transpose(0, 1) # B x 1 x N -> 1 x B x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        concat_input = torch.cat((rnn_output, context), 2)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Squeeze the time dimension
        output = output.squeeze(0)

        # Apply log_softmax
        output = self.log_softmax(output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

    def forward(self, context_vector, encoder_outputs, max_length):

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

        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        # Unfold the decoder RNN on the time dimension
        for t in range(max_length):
            decoder_output, decoder_hidden, attn_weights = \
                self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[t] = decoder_output
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.transpose(0, 1)

        # Return final output, hidden state, and attention weights (for visualization)
        return decoder_outputs, decoder_hidden, attn_weights

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            hidden = hidden.cuda()
        return hidden
