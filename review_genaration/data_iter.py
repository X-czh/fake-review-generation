import math
import random
import pandas as pd
import pickle as pkl

import torch
from torch.utils.data import Dataset, DataLoader

from utils import pad_sequence


class GenDataset(Dataset):

    def __init__(self, vocab_file, data_file, max_seq_len):
        self.max_seq_len = max_seq_len
        self.data_lis = self.read_file(vocab_file, data_file, max_seq_len)
        self.data_num = len(self.data_lis)

    def __getitem__(self, index):
        seq_tensor = torch.tensor(self.data_lis[index], dtype=torch.int64)

        # 0 <SOS_token> is prepended to seq_tensor as start symbol
        data = torch.cat([torch.zeros(1, dtype=torch.int64), seq_tensor])
        target = torch.cat([seq_tensor, torch.zeros(1, dtype=torch.int64)])

        return data, target

    def __len__(self):
        return self.data_num

    def read_file(self, vocab_file, data_file, max_seq_len):
        with open(vocab_file, 'rb') as f:
            lang = pkl.load(f)
        df = pd.read_csv(data_file, delimiter='\t')
        lis = []
        for line in df['text'][:1000]:
            l = [lang.word2index(s) for s in line.split(' ')]
            l = pad_sequence(l, max_seq_len)
            lis.append(l)
        return lis


class DisDataset(Dataset):

    def __init__(self, vocab_file, real_data_file, fake_data_file, max_seq_len):
        self.max_seq_len = max_seq_len
        real_data_lis = self.read_file(vocab_file, real_data_file, max_seq_len)
        fake_data_lis = self.read_file(vocab_file, fake_data_file, max_seq_len)
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        data, label = self.pairs[index]
        data = torch.tensor(data, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)
        return data, label

    def read_file(self, vocab_file, data_file, max_seq_len):
        with open(vocab_file, 'rb') as f:
            lang = pkl.load(f)
        df = pd.read_csv(data_file, delimiter='\t')
        df = df.sample(frac=1).reset_index(drop=True)
        lis = []
        for line in df['text'][:10000]:
            l = [lang.word2index(s) for s in line.split(' ')]
            l = pad_sequence(l, max_seq_len) # pad sequence for CNN
            lis.append(l)
        return lis


def getGenDataIter(vocab_file, data_file, batch_size, max_seq_len):
    dataset = GenDataset(vocab_file, data_file, max_seq_len)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    return dataloader


def getDisDataIter(vocab_file, real_data_file, fake_data_file, batch_size, max_seq_len):
    dataset = DisDataset(vocab_file, real_data_file, fake_data_file, max_seq_len)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    return dataloader
