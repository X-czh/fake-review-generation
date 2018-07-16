import numpy as np
import pandas as pd
from collections import OrderedDict


SOS_token = 0 # start of sequence
UNK_token = 1 # unknown


def pad_sequence(seq, max_seq_len):
    """
    Pad sequence seq to max_seq_len on its right side
    """
    seq.extend([0 for i in range(max_seq_len - len(seq))])
    return seq


class Lang:
    """
    Class for building dictionary of corpus
    """

    def __init__(self, vocab_size, data_file):
        self._vocab_size = vocab_size
        self._word2index = OrderedDict()
        self._index2word = OrderedDict()
        self._vocab = {"<SOS>": 100000001, "<UNK>": 100000000}
        self.read_file(data_file)
        self.create_dictionary()

    def word2index(self, word):
        if word in self._word2index:
            index = self._word2index[word]
            if index > self._vocab_size:
                return UNK_token
            else:
                return index
        else:
            return UNK_token
    
    def index2word(self, index):
        assert index in self._index2word
        return self._index2word[index]

    def read_file(self, data_file):
        df = pd.read_csv(data_file, delimiter='\t')
        for sent in df['text']:
            self.addSentence(sent)

    def addSentence(self, sent):
        for word in sent.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word in self._vocab:
            self._vocab[word] += 1
        else:
            self._vocab[word] = 1

    def create_dictionary(self):
        tokens = list(self._vocab.keys())
        freqs = list(self._vocab.values())
        sidx = np.argsort(freqs)[::-1]
        self._word2index = OrderedDict([(tokens[s], i) for i, s in enumerate(sidx)])
        self._index2word = OrderedDict([(i, tokens[s]) for i, s in enumerate(sidx)])
