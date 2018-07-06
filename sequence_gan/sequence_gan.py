import argparse
import os
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import get_dataloader
from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout


# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--data-path', type=str, default='/home/x-czh/data_set', metavar='PATH',
                    help='data path (default: /home/x-czh/data_set)')
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--milestones', type=str, default='0', metavar='M',
                    help='milestones to adjust learning rate, ints split by "-" (default: "0")')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--workers', type=int, default=2, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from latest checkpoint')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 200
GENERATED_NUM = 10000
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 5000
PRE_EPOCH_NUM = 120

g_steps = 1
d_steps = 2
k = 3

# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 20

# Discriminator Parameters
d_embed_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

d_dropout_prob = 0.75
d_num_classed = 2


def generate_samples(generator, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = generator.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


def train_generator_MLE(generator, dataloader, criterion, optimizer, epochs, use_cuda):
    """
    Pre-train the generator with maximum likelihood estimation 
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        total_loss = 0.
        total_words = 0.
        for data, target in enumerate(dataloader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data.requires_grad = True
            target.requires_grad = True
            pred = generator(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return math.exp(total_loss / total_words)

def train_generator_PG(generator):
    """
    """
    pass

def train_discriminator(discriminator, criterion, dataloader, use_cuda):
    total_loss = 0.
    total_words = 0.
    for data, target in dataloader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        target = target.view(-1)
        pred = discriminator(data)
        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return math.exp(total_loss / total_words)

def adversial_train(g_steps, d_steps, k, model, update_rate):
    rollout = Rollout(generator,update_rate)
    for i in range(g_steps):
        pass
    for i in range(d_steps):
        for j in range(k):
            pass

if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Set models, criteria and optimizers
    generator = Generator(args.vocab_size, args.embedding_dim) #TODO
    discriminator = Discriminator() #TODO
    generator_criterion = nn.NLLLoss()
    discriminator_criterion = nn.NLLLoss()
    generator_optimizer = optim.Adam(generator.parameters())
    discriminator_optimizer = optim.Adam(discriminator.parameters())
    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        generator_criterion = generator_criterion.cuda()
        discriminator_criterion = discriminator_criterion.cuda()

    # Pre-train the generator with MLE
    print('Start Pre-training Generator with MLE...')
    train_generator_MLE(generator, generator_dataloader, generator_criterion, generator_optimizer)

    # Set the roll-out policy to be the generator
    rollout_model = generator

    # Generate negative samples

    # Pre-train the discrimiator
    print('Start Pre-training Discrimiator...')
    train_discriminator(discriminator)

    # Adversarial training
    print('Start Adversarial Training...')
    adversarial_train(rollout_model, update_rate)
