import argparse
import pandas as pd
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import Lang
from data_iter import getGenDataIter, getDisDataIter
from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from loss import PGLoss


# Arguemnts
parser = argparse.ArgumentParser(description='Review Generation')
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data_path', type=str, default='/scratch/zc807/review_generation/', metavar='PATH',
                    help='data path to save files (default: /scratch/zc807/review_generation/)')
parser.add_argument('--rounds', type=int, default=30, metavar='N',
                    help='rounds of adversarial training (default: 30)')
parser.add_argument('--g_pretrain_steps', type=int, default=20, metavar='N',
                    help='steps of pre-training of generators (default: 20)')
parser.add_argument('--d_pretrain_steps', type=int, default=10, metavar='N',
                    help='steps of pre-training of discriminators (default: 10)')
parser.add_argument('--g_steps', type=int, default=1, metavar='N',
                    help='steps of generator updates in one round of adverarial training (default: 1)')
parser.add_argument('--d_steps', type=int, default=3, metavar='N',
                    help='steps of discriminator updates in one round of adverarial training (default: 5)')
parser.add_argument('--gk_epochs', type=int, default=1, metavar='N',
                    help='epochs of generator updates in one step of generate update (default: 1)')
parser.add_argument('--dk_epochs', type=int, default=3, metavar='N',
                    help='epochs of discriminator updates in one step of discriminator update (default: 3)')
parser.add_argument('--update_rate', type=float, default=0.8, metavar='UR',
                    help='update rate of roll-out model (default: 0.8)')
parser.add_argument('--n_rollout', type=int, default=16, metavar='N',
                    help='number of roll-out (default: 16)')
parser.add_argument('--vocab_size', type=int, default=10000, metavar='N',
                    help='vocabulary size (default: 10000)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--n_samples', type=int, default=10000, metavar='N',
                    help='number of samples gerenated per time (default: 10000)')
parser.add_argument('--gen_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of generator optimizer (default: 1e-3)')
parser.add_argument('--dis_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of discriminator optimizer (default: 1e-3)')
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
g_seq_len = 30


# Discriminator Parameters
d_num_class = 2
d_embed_dim = 300
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_dropout_prob = 0.5


def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    texts = [' '.join([str(s) for s in sample]) for sample in samples]
    df = pd.DataFrame(texts, columns=['text'])
    df.to_csv(output_file, sep='\t', encoding='utf-8')


def train_generator_MLE(gen, data_iter, criterion, optimizer, epochs, 
        gen_pretrain_train_loss, args):
    """
    Train generator with MLE
    """
    for epoch in range(epochs):
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
        gen_pretrain_train_loss.append(avg_loss)


def train_generator_PG(gen, dis, rollout, pg_loss, optimizer, epochs, args):
    """
    Train generator with the guidance of policy gradient
    """
    for epoch in range(epochs):
        # construct the input to the genrator, add zeros before samples and delete the last column
        samples = generator.sample(args.batch_size, g_seq_len)
        zeros = torch.zeros(args.batch_size, 1, dtype=torch.int64)
        if samples.is_cuda:
            zeros = zeros.cuda()
        inputs = torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous()
        targets = samples.data.contiguous().view((-1,))

        # calculate the reward
        rewards = torch.tensor(rollout.get_reward(samples, args.n_rollout, dis))
        if args.cuda:
            rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
        
        # update generator
        output = gen(inputs)
        loss = pg_loss(output, targets, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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


def train_discriminator(dis, gen, criterion, optimizer, epochs, 
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Train discriminator
    """
    generate_samples(gen, args.batch_size, args.n_samples, NEGATIVE_FILE)
    data_iter = getDisDataIter(VOCAB_FILE, POSITIVE_FILE, NEGATIVE_FILE, args.batch_size, g_seq_len, sample=True)
    for epoch in range(epochs):
        correct = 0
        total_loss = 0.
        for data, target in data_iter:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            output = dis(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            loss = criterion(output, target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(data_iter)
        acc = correct.item() / len(data_iter.dataset)
        print("Epoch {}, train loss: {:.5f}, train acc: {:.3f}".format(epoch, avg_loss, acc))
        dis_adversarial_train_loss.append(avg_loss)
        dis_adversarial_train_acc.append(acc)


def eval_discriminator(model, data_iter, criterion, args):
    """
    Evaluate discriminator, dropout is enabled
    """
    correct = 0
    total_loss = 0.
    for data, target in data_iter:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        output = model(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        loss = criterion(output, target)
        total_loss += loss.item()
    avg_loss = total_loss / len(data_iter)
    acc = correct.item() / len(data_iter.dataset)
    return avg_loss, acc


def adversarial_train(gen, dis, rollout, pg_loss, nll_loss, gen_optimizer, dis_optimizer, 
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Adversarially train generator and discriminator
    """
    # train generator for g_steps
    print("#Train generator")
    for i in range(args.g_steps):
        print("##G-Step {}".format(i))
        train_generator_PG(gen, dis, rollout, pg_loss, gen_optimizer, args.gk_epochs, args)

    # train discriminator for d_steps
    print("#Train discriminator")
    for i in range(args.d_steps):
        print("##D-Step {}".format(i))
        train_discriminator(dis, gen, nll_loss, dis_optimizer, args.dk_epochs, 
            dis_adversarial_train_loss, dis_adversarial_train_acc, args)

    # update roll-out model
    rollout.update_params()


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
    NEGATIVE_FILE = args.data_path + NEGATIVE_FILE
    EVAL_FILE = args.data_path + EVAL_FILE

    # Set models, criteria, optimizers
    generator = Generator(args.vocab_size, g_embed_dim, g_hidden_dim, args.cuda)
    discriminator = Discriminator(d_num_class, args.vocab_size, d_embed_dim, d_filter_sizes, d_num_filters, d_dropout_prob)
    nll_loss = nn.NLLLoss()
    pg_loss = PGLoss()
    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        nll_loss = nll_loss.cuda()
        pg_loss = pg_loss.cuda()
        cudnn.benchmark = True
    gen_optimizer = optim.Adam(params=generator.parameters(), lr=args.gen_lr)
    dis_optimizer = optim.SGD(params=discriminator.parameters(), lr=args.dis_lr)

    # Container of experiment data
    gen_pretrain_train_loss = []
    gen_pretrain_eval_loss = []
    dis_pretrain_train_loss = []
    dis_pretrain_train_acc = []
    dis_pretrain_eval_loss = []
    dis_pretrain_eval_acc = []
    gen_adversarial_eval_loss = []
    dis_adversarial_train_loss = []
    dis_adversarial_train_acc = []
    dis_adversarial_eval_loss = []
    dis_adversarial_eval_acc = []

    # Read real data and build vocab dictionary
    lang = Lang(args.vocab_size, POSITIVE_FILE)
    with open(VOCAB_FILE, 'wb') as f:
        pkl.dump(lang, f, protocol=pkl.HIGHEST_PROTOCOL) 

    # Pre-train generator using MLE
    print('#####################################################')
    print('Start pre-training generator with MLE...')
    print('#####################################################\n')
    gen_data_iter = getGenDataIter(VOCAB_FILE, POSITIVE_FILE, args.batch_size, g_seq_len)
    for i in range(args.g_pretrain_steps):
        print("G-Step {}".format(i))
        train_generator_MLE(generator, gen_data_iter, nll_loss, 
            gen_optimizer, args.gk_epochs, gen_pretrain_train_loss, args)
        eval_iter = getGenDataIter(VOCAB_FILE, EVAL_FILE, args.batch_size, g_seq_len)
        gen_loss = eval_generator(generator, eval_iter, nll_loss, args)
        gen_pretrain_eval_loss.append(gen_loss)
        print("eval loss: {:.5f}\n".format(gen_loss))
    print('#####################################################\n\n')

    # Pre-train discriminator
    print('#####################################################')
    print('Start pre-training discriminator...')
    print('#####################################################\n')
    for i in range(args.d_pretrain_steps):
        print("D-Step {}".format(i))
        train_discriminator(discriminator, generator, nll_loss, 
            dis_optimizer, args.dk_epochs, dis_adversarial_train_loss, dis_adversarial_train_acc, args)
        generate_samples(generator, args.batch_size, args.n_samples, NEGATIVE_FILE)
        eval_iter = getDisDataIter(VOCAB_FILE, EVAL_FILE, NEGATIVE_FILE, args.batch_size, g_seq_len)
        dis_loss, dis_acc = eval_discriminator(discriminator, eval_iter, nll_loss, args)
        dis_pretrain_eval_loss.append(dis_loss)
        dis_pretrain_eval_acc.append(dis_acc)
        print("eval loss: {:.5f}, eval acc: {:.3f}\n".format(dis_loss, dis_acc))
    print('#####################################################\n\n')

    # Adversarial training
    print('#####################################################')
    print('Start adversarial training...')
    print('#####################################################\n')
    rollout = Rollout(generator, args.update_rate)
    for i in range(args.rounds):
        print("Round {}".format(i))
        adversarial_train(generator, discriminator, rollout, 
            pg_loss, nll_loss, gen_optimizer, dis_optimizer, 
            dis_adversarial_train_loss, dis_adversarial_train_acc, args)
        generate_samples(generator, args.batch_size, args.n_samples, NEGATIVE_FILE)
        gen_eval_iter = getGenDataIter(VOCAB_FILE, NEGATIVE_FILE, args.batch_size, g_seq_len)
        dis_eval_iter = getDisDataIter(VOCAB_FILE, EVAL_FILE, NEGATIVE_FILE, args.batch_size, g_seq_len)
        gen_loss = eval_generator(generator, gen_eval_iter, nll_loss, args)
        gen_adversarial_eval_loss.append(gen_loss)
        dis_loss, dis_acc = eval_discriminator(discriminator, dis_eval_iter, nll_loss, args)
        dis_adversarial_eval_loss.append(dis_loss)
        dis_adversarial_eval_acc.append(dis_acc)
        print("gen eval loss: {:.5f}, dis eval loss: {:.5f}, dis eval acc: {:.3f}\n"
            .format(gen_loss, dis_loss, dis_acc))

    # Save experiment data
    with open(args.data_path + 'experiment.pkl', 'wb') as f:
        pkl.dump(
            (gen_pretrain_train_loss,
                gen_pretrain_eval_loss,
                dis_pretrain_train_loss,
                dis_pretrain_train_acc,
                dis_pretrain_eval_loss,
                dis_pretrain_eval_acc,
                gen_adversarial_eval_loss,
                dis_adversarial_train_loss,
                dis_adversarial_train_acc,
                dis_adversarial_eval_loss,
                dis_adversarial_eval_acc),
            f,
            protocol=pkl.HIGHEST_PROTOCOL
        )
