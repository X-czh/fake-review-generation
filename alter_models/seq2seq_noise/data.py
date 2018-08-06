import math
import random
import torch
from utils import PAD_token, UNK_token, SOS_token, EOS_token, pad_sequence

class Lang:
    def __init__(self, lang_load):
        self.word2index = lang_load[0]
        self.word2count = lang_load[1]
        self.index2word = lang_load[2]
        self.n_words = lang_load[3]

def indexesFromSentence(lang, sentence, vocab_size):
    result = []
    for word in sentence.split(' '):
        if word in lang.word2index and lang.word2index[word] < vocab_size:
            result.append(lang.word2index[word])
        else:
            result.append(UNK_token)
    result.append(EOS_token)
    return result

def variableFromSentence(lang, sentence, vocab_size, use_cuda):
    indexes = indexesFromSentence(lang, sentence, vocab_size)
    result = torch.LongTensor(indexes).view(-1, 1)
    if use_cuda:
        return result.cuda()
    else:
        return result

class DataIter:
    def __init__(self, pairs, lang, vocab_size, batch_size, use_cuda):
        self.lang = lang
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.pairs = pairs
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = math.floor(self.data_num / self.batch_size)
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_num - self.idx < self.batch_size:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        
        input_seqs = []
        target_seqs = []

        # Choose pairs
        for i in index:
            pair = self.pairs[i]
            input_seqs.append(indexesFromSentence(self.lang, pair[0], self.vocab_size))
            target_seqs.append(indexesFromSentence(self.lang, pair[1], self.vocab_size))

        # Zip into pairs, sort by length (descending), unzip
        seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)

        # For input and target sequences, get array of lengths and pad with 0s to max length
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_sequence(s, max(input_lengths)) for s in input_seqs]
        target_lengths = [len(s) for s in target_seqs]
        target_padded = [pad_sequence(s, max(target_lengths)) for s in target_seqs]

        # Turn padded arrays into (batch x seq) tensors, transpose into (seq x batch)
        input_tensor = torch.LongTensor(input_padded).transpose(0, 1)
        target_tensor = torch.LongTensor(target_padded).transpose(0, 1)

        if self.use_cuda:
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()

        self.idx += self.batch_size

        return input_tensor, input_lengths, target_tensor, target_lengths

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)
