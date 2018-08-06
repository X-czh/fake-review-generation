import argparse
import random
import numpy as np
import pandas as pd
import pickle as pkl
from collections import OrderedDict


###############################################
# Preprocessing settings
###############################################

parser = argparse.ArgumentParser(description='Data Preprocessing')
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data-path', type=str, default='/scratch/zc807/data', metavar='PATH',
                    help='data path (default: /scratch/zc807/data)')
parser.add_argument('--save-data-path', type=str, default='/scratch/zc807/attn2seq', metavar='PATH',
                    help='data path to save pairs.pkl and lang.pkl (default: /scratch/zc807/attn2seq)')


###############################################
# Core classes and functions
###############################################

class Lang:
    def __init__(self, name):
        self.name = name
        self.n_words = 4  # Count PAD, UNK, SOS and EOS
        self.word2index = None
        self.word2count = {"<PAD>": 100000003, "<SOS>": 100000002, "<EOS>": 100000001, "<UNK>": 100000000}
        self.index2word = None

    def addSentence(self, sent):
        for word in sent.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word in self.word2count:
            self.word2count[word] += 1
        else:
            self.n_words += 1
            self.word2count[word] = 1

    def create_dictionary(self):
        tokens = list(self.word2count.keys())
        freqs = list(self.word2count.values())
        sidx = np.argsort(freqs)[::-1]
        self.word2index = OrderedDict([(tokens[s], i) for i, s in enumerate(sidx)])
        self.index2word = OrderedDict([(i, tokens[s]) for i, s in enumerate(sidx)])

def readData(data_path):
    print("Reading lines...")

    # Read the file
    df_train = pd.read_csv(data_path + '/train.csv', delimiter='\t')
    df_test = pd.read_csv(data_path + '/test.csv', delimiter='\t')

    # Construct train pairs
    train_reviews = df_train['text']
    train_categories = df_train['categories']
    train_stars = df_train['stars']
    train_context = [str(train_stars[i]) + ' ' + train_categories[i] \
        for i in range(len(train_categories))]
    train_pairs = list(zip(train_context, train_reviews))

    # Construct test pairs
    test_reviews = df_test['text']
    test_categories = df_test['categories']
    test_stars = df_test['stars']
    test_context = [str(test_stars[i]) + ' ' + test_categories[i] \
        for i in range(len(test_categories))]
    test_pairs = list(zip(test_context, test_reviews))

    return train_pairs, test_pairs

def prepareData(data_path):
    lang = Lang('Yelp Reviews')
    train_pairs, test_pairs = readData(data_path)
    print("Read %s training sentence pairs" % len(train_pairs))
    print("Read %s testing sentence pairs" % len(test_pairs))

    print("Counting words and constructing training pairs...")
    for pair in train_pairs:
        lang.addSentence(pair[0])
        lang.addSentence(pair[1])
    lang.create_dictionary()
    print("Counted words in training sentences:")
    print(lang.name, lang.n_words)

    return lang, train_pairs, test_pairs

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.hpc:
        args.data_path = '../data'
        args.save_data_path = '.'
    
    print("hpc mode: {}".format(args.hpc))
    lang, train_pairs, test_pairs = prepareData(args.data_path)
    lang_tuple = (lang.word2index, lang.word2count, lang.index2word, lang.n_words)

    with open(args.save_data_path + "/pairs.pkl", 'wb') as f:
        pkl.dump((train_pairs, test_pairs), f, protocol=pkl.HIGHEST_PROTOCOL) 
    with open(args.save_data_path + "/lang.pkl", 'wb') as f:
        pkl.dump(lang_tuple, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    print("Example training sentence pairs:")
    print(random.choice(train_pairs))
    print("Example testing sentence pairs:")
    print(random.choice(test_pairs))
