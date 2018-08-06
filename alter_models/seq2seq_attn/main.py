import argparse
import os
import time
import random
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from model import EncoderBiRNN, AttnDecoderRNN
from data import Lang, DataIter, variableFromSentence
from metric import score, multi_score
from utils import PAD_token, UNK_token, SOS_token, EOS_token, asMinutes, timeSince, showPlot


###############################################
# Training settings
###############################################

parser = argparse.ArgumentParser(description='seq2seq')
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data_path', type=str, default='/scratch/zc807/seq2seq_attn', metavar='PATH',
                    help='data path of pairs.pkl and lang.pkl (default: /scratch/zc807/seq2seq_attn)')
parser.add_argument('--save_data_path', type=str, default='/scratch/zc807/seq2seq_attn', metavar='PATH',
                    help='data path to save model parameters (default: /scratch/zc807/seq2seq_attn)')
parser.add_argument('--resume', type=str, default='', metavar='PATH',
                    help='data path to load checkpoint for resuming (default: none)')
parser.add_argument('--metric', type=str, default='MULTI', metavar='METRIC',
                    help='metric to use (default: MULTI; ROUGE, BLEU and BLEU_clip available)')
parser.add_argument('--vocab_size', type=int, default='15000', metavar='N',
                    help='vocab size (default: 15000)')
parser.add_argument('--hidden_size', type=int, default='500', metavar='N',
                    help='hidden size (default: 500)')
parser.add_argument('--batch_size', type=int, default='64', metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--n_layers', type=int, default='2', metavar='N',
                    help='number of stacked layers of RNNs (default: 2)')
parser.add_argument('--dropout', type=float, default='0.1', metavar='DR',
                    help='dropout_prob for stacked RNNs (default: 0.1)')
parser.add_argument('--attn', choices=['dot', 'general', 'concat'], default='general', metavar='ATTN',
                    help='attention model for decoder to use (default: general)')
parser.add_argument('--n_epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--n_batches', type=int, default=0, metavar='N',
                    help='number of batches to train (default: 0 (disabled)), for testing only')
parser.add_argument('--print_every', type=int, default='100', metavar='N',
                    help='print every (default: 100) batches')
parser.add_argument('--plot_every', type=int, default='10', metavar='N',
                    help='plot every (default: 10) batches')
parser.add_argument('--save_every', type=int, default='10', metavar='N',
                    help='save checkpoint every (default: 10) epochs')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, metavar='TFR',
                    help='teacher forcing ratio (default: 0.5)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=0, metavar='WD',
                    help='weight decay (L2 penalty) (default: 0)')
parser.add_argument('--clip', type=float, default=0, metavar='CLIP',
                    help='gradient clip threshold (default: 0 (disabled)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1997, metavar='S',
                    help='random seed (default: 1997)')
parser.set_defaults(max_length=40)


###############################################
# Training
###############################################

 
def train(input_tensor, input_lengths, target_tensor, target_lengths, 
        encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, args):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    loss = 0

    # Run EncoderRNN on inputs
    encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)

    # Concatenate bidirectional encoder hidden as context vector for decoder
    context_vector = torch.cat([
        encoder_hidden[0:encoder_hidden.size(0):2], 
        encoder_hidden[1:encoder_hidden.size(0):2]
        ], 2)

    # Prepare tensor for decoder on time_step_0
    batch_size = context_vector.size(1)
    decoder_input = torch.LongTensor([[SOS_token] * batch_size])

    # Pass the context vector
    decoder_hidden = context_vector

    max_target_length = max(target_lengths)
    decoder_outputs = torch.zeros(
        max_target_length, 
        batch_size, 
        decoder.output_size
    ) # (time_steps, batch_size, vocab_size)

    if decoder.use_cuda:
        decoder_input = decoder_input.cuda()
        decoder_outputs = decoder_outputs.cuda()

    use_teacher_forcing = True if random.random() > args.teacher_forcing_ratio else False

    # Unfold the decoder RNN on the time dimension
    for t in range(max_target_length):
        decoder_output, decoder_hidden, _ = decoder.forward_step(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_outputs[t] = decoder_output
        if use_teacher_forcing:
            decoder_input = target_tensor[t].unsqueeze(0)
        else:
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.transpose(0, 1)

    loss += criterion(decoder_outputs.view(-1, args.vocab_size), target_tensor.contiguous().view(-1))
    loss.backward()

    # Clip gradient
    if args.clip > 0:
        nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def trainEpochs(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, dataiter, args):
    n_epochs = args.n_epochs
    print_every = args.print_every
    plot_every = args.plot_every

    start = time.time()
    batch_i = 0
    n_batches = n_epochs * len(dataiter)
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for epoch in range(args.n_epochs):

        for input_tensor, input_lengths, target_tensor, target_lengths in dataiter:
            batch_i += 1

            loss = train(input_tensor, input_lengths, target_tensor, target_lengths, 
                encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, args)
            print_loss_total += loss
            plot_loss_total += loss

            if batch_i % args.print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, batch_i / n_batches),
                                            batch_i, batch_i / n_batches * 100, print_loss_avg))

            if batch_i % args.plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if (epoch + 1) % args.save_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder_optim_state': encoder_optimizer.state_dict(),
                    'decoder_optim_state': decoder_optimizer.state_dict(),
                }
                torch.save(checkpoint, args.save_data_path + "/epoch{}_checkpoint.pt".format(epoch))

            # for testing only
            if args.n_batches > 0 and batch_i == args.n_batches:
                break

        dataiter.reset()
        
        print("Epoch {}/{} finished".format(epoch, args.n_epochs - 1))

    showPlot(plot_losses, args)


###############################################
# Evaluation
###############################################

def evaluate(encoder, decoder, sentence, input_lang, output_lang, args):
    use_cuda = args.cuda
    max_length = args.max_length

    with torch.no_grad():
        input_tensor = variableFromSentence(input_lang, sentence, args.vocab_size, use_cuda)
        input_lengths = [input_tensor.size(0)]
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)

        decoder_input = torch.LongTensor([[SOS_token]])  # SOS
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        # Concatenate bidirectional encoder hidden as context vector for decoder
        context_vector = torch.cat([
            encoder_hidden[0:encoder_hidden.size(0):2], 
            encoder_hidden[1:encoder_hidden.size(0):2]
            ], 2)
        decoder_hidden = context_vector

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, _ = decoder.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.item()
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[ni])

            decoder_input = torch.LongTensor([[ni]])
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, args, n=10):
    for i in range(n):
        print('PAIR {}'.format(i+1))
        pair = random.choice(pairs)
        for j in range(5):
            print('>', pair[0])
            print('=', pair[1])
            output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang, args)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

def evaluateTestingPairs(encoder, decoder, pairs, input_lang, output_lang, args):
    score_short = 0
    score_long = 0
    list_cand_short = []
    list_ref_short = []
    list_cand_long = []
    list_ref_long = []

    print("Evaluating {} testing sentences...".format(len(pairs)))
    
    for pair in pairs:
        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang, args)
        output_sentence = ' '.join(output_words)
        sent_length = len(pair[1].split(' '))
        if sent_length > (15):
            list_cand_long.append(output_sentence)
            list_ref_long.append(pair[1])
        else:
            list_cand_short.append(output_sentence)
            list_ref_short.append(pair[1])

    print("Num of short sentences (length <= 15):", len(list_cand_short))
    if len(list_cand_short) > 0:
        if args.metric == "MULTI":
            score_short_rouge1, score_short_rouge2, score_short_bleu, score_short_bleu_clip = \
                multi_score(list_cand_short, list_ref_short)
            print("score for short sentnces (length <= 15):")
            print("ROUGE1:", score_short_rouge1)
            print("ROUGE2:", score_short_rouge2)
            print("BLEU:", score_short_bleu)
            print("BLEU_CLIP:", score_short_bleu_clip)
            print()
        else:
            score_short = score(list_cand_short, list_ref_short, args.metric)
            print("{} score for short sentnces (length <= 15): {}".format(args.metric, score_short))

    print("Num of long sentences (length > 15):", len(list_cand_long))
    if len(list_cand_long) > 0:
        if args.metric == "MULTI":
            score_long_rouge1, score_long_rouge2, score_long_bleu, score_long_bleu_clip = \
                multi_score(list_cand_long, list_ref_long)
            print("score for long sentnces (length > 15):")
            print("ROUGE1:", score_long_rouge1)
            print("ROUGE2:", score_long_rouge2)
            print("BLEU:", score_long_bleu)
            print("BLEU_CLIP:", score_long_bleu_clip)
            print()
        else:
            score_long = score(list_cand_long, list_ref_long, args.metric)
            print("{} score for long sentnces (length > 15): {}".format(args.metric, score_long))

    get_score_overall = lambda score_short, score_long: \
        (score_short * len(list_cand_short) + score_long * len(list_cand_long)) \
        / (len(list_cand_short) + len(list_cand_long))
    if args.metric == "MULTI":
            score_overall_rouge1 = get_score_overall(score_short_rouge1, score_long_rouge1)
            score_overall_rouge2 = get_score_overall(score_short_rouge2, score_long_rouge2)
            score_overall_bleu = get_score_overall(score_short_bleu, score_long_bleu)
            score_overall_bleu_clip = get_score_overall(score_short_bleu_clip, score_long_bleu_clip)
            print("Overall:")
            print("ROUGE1:", score_overall_rouge1)
            print("ROUGE2:", score_overall_rouge2)
            print("BLEU:", score_overall_bleu)
            print("BLEU_CLIP:", score_overall_bleu_clip)
            print()
    else:
        score_overall = get_score_overall(score_short, score_long)
        print("Overall {} score: {}".format(args.metric, score_overall))


###############################################
# Main
###############################################

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.hpc:
        args.data_path = '.'
        args.save_data_path = '.'
    if args.cuda:
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    # Print settings
    print("hpc mode: {}".format(args.hpc))
    print("data-path: {}".format(args.data_path))
    print("save-data-path: {}".format(args.save_data_path))
    print("resume: {}".format(args.resume))
    print("metric: {}".format(args.metric))
    print("vocab-size: {}".format(args.vocab_size))
    print("hidden-size: {}".format(args.hidden_size))
    print("batch-size: {}".format(args.batch_size))
    print("n-layers: {}".format(args.n_layers))
    print("dropout: {}".format(args.dropout))
    print("attn: {}".format(args.attn))
    print("n-epochs: {}".format(args.n_epochs))
    print("n-batches (testing only): {}".format(args.n_batches))
    print("print-every: {}".format(args.print_every))
    print("plot-every: {}".format(args.plot_every))
    print("save-every: {}".format(args.save_every))
    print("teacher forcing ratio: {}".format(args.teacher_forcing_ratio))
    print("lr: {}".format(args.lr))
    print("weight-decay: {}".format(args.weight_decay))
    print("clip: {}".format(args.clip))
    print("use cuda: {}".format(args.cuda))
    print("seed: {}".format(args.seed))

    # Load pairs.pkl and lang.pkl
    with open(args.data_path + "/pairs.pkl", 'rb') as f:
        (train_pairs, test_pairs) = pkl.load(f)
    with open(args.data_path + "/lang.pkl", 'rb') as f:
        lang_tuple = pkl.load(f)
    lang = Lang(lang_tuple)

    # Prepare dataloader for training
    train_dataiter = DataIter(train_pairs, lang, args.vocab_size, args.batch_size, args.cuda)

    # Set encoder and decoder
    encoder = EncoderBiRNN(args.vocab_size, args.hidden_size, args.n_layers, args.dropout)
    decoder = AttnDecoderRNN(args.attn, args.hidden_size * 2, args.vocab_size, 
        args.n_layers, args.dropout, args.cuda)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    # Set optimizer and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            encoder_optimizer.load_state_dict(checkpoint['encoder_optim_state'])
            decoder_optimizer.load_state_dict(checkpoint['encoder_optim_state'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Train and evalute
    print("\nStart")
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, decoder, train_pairs, lang, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, decoder, test_pairs, lang, lang, args)
    trainEpochs(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, train_dataiter, args)
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, decoder, train_pairs, lang, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, decoder, test_pairs, lang, lang, args)
    evaluateTestingPairs(encoder, decoder, test_pairs, lang, lang, args)
    print("Finished\n")
