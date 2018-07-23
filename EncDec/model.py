import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, use_cuda):
        super(EncoderRNN, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        self.gru.flatten_parameters()
        embed = self.embedding(inputs)
        output, hidden = self.gru(embed, hidden)
        return output, hidden

    def initHidden(self):
        result = torch.zeros(1, 1, self.hidden_size)
        if self.use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, use_cuda):
        super(DecoderRNN, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.log_softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = torch.zeros(1, 1, self.hidden_size)
        if self.use_cuda:
            return result.cuda()
        else:
            return result
