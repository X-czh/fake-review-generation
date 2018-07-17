import argparse
import pandas as pd
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import Lang
from data_iter import getGenDataIter
from generator import Generator


# Arguemnts
parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data_path', type=str, default='/scratch/zc807/baseline/', metavar='PATH',
                    help='data path to save files (default: /scratch/zc807/baseline/)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='epochs of training of generators (default: 120)')
parser.add_argument('--vocab_size', type=int, default=10000, metavar='N',
                    help='vocabulary size (default: 10000)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--gen_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of generator optimizer (default: 1e-3)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


# Files
VOCAB_FILE = 'vocab.pkl'
POSITIVE_FILE = '../data/train.csv'
EVAL_FILE = '../data/val.csv'
NEGATIVE_FILE = 'gene.csv'


# Genrator Parameters
g_embed_dim = 300
g_hidden_dim = 300
g_seq_len = 60


def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    texts = [' '.join([str(s) for s in sample]) for sample in samples]
    df = pd.DataFrame(texts, columns=['text'])
    df.to_csv(output_file, sep='\t', encoding='utf-8')


def train_generator_MLE(gen, data_iter, criterion, optimizer, epoch, 
        train_loss, args):
    """
    Train generator with MLE
    """
    total_loss = 0.
    for data, target in data_iter:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        output = gen(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(data_iter)
    print("Epoch {}, train loss: {:.5f}".format(epoch, avg_loss))
    train_loss.append(avg_loss)


def eval_generator(model, data_iter, criterion, args):
    """
    Evaluate generator with NLL
    """
    total_loss = 0.
    for data, target in data_iter:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
    avg_loss = total_loss / len(data_iter)
    return avg_loss


if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if not args.hpc:
        args.data_path = ''
    VOCAB_FILE = args.data_path + VOCAB_FILE
    POSITIVE_FILE = args.data_path + POSITIVE_FILE
    EVAL_FILE = args.data_path + EVAL_FILE

    # Set models, criteria, optimizers
    generator = Generator(args.vocab_size, g_embed_dim, g_hidden_dim, args.cuda)
    nll_loss = nn.NLLLoss()
    gen_optimizer = optim.Adam(params=generator.parameters(), lr=args.gen_lr)
    if args.cuda:
        generator = generator.cuda()
        nll_loss = nll_loss.cuda()
        cudnn.benchmark = True

    # Container of experiment data
    train_loss = []
    eval_loss = []

    # Read real data and build vocab dictionary
    lang = Lang(args.vocab_size, POSITIVE_FILE)
    with open(VOCAB_FILE, 'wb') as f:
        pkl.dump(lang, f, protocol=pkl.HIGHEST_PROTOCOL) 

    # Pre-train generator using MLE
    print('#####################################################')
    print('Start pre-training generator with MLE...')
    print('#####################################################\n')
    gen_data_iter = getGenDataIter(VOCAB_FILE, POSITIVE_FILE, args.batch_size, g_seq_len)
    for epoch in range(args.epochs):
        train_generator_MLE(generator, gen_data_iter, nll_loss, 
            gen_optimizer, epoch, train_loss, args)
        eval_iter = getGenDataIter(VOCAB_FILE, EVAL_FILE, args.batch_size, g_seq_len)
        gen_loss = eval_generator(generator, eval_iter, nll_loss, args)
        eval_loss.append(gen_loss)
        print("eval loss: {:.5f}\n".format(gen_loss))
    print('#####################################################\n\n')

    # Generate samples
    generate_samples(generator, args.batch_size, args.n_samples, NEGATIVE_FILE)

    # Save experiment data
    with open(args.data_path + 'experiment.pkl', 'wb') as f:
        pkl.dump((train_loss, eval_loss),
            f, protocol=pkl.HIGHEST_PROTOCOL
        )
