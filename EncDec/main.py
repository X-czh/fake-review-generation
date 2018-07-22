import argparse
import time
import random
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from model import EncoderRNN, DecoderRNN
from metric import score, multi_score
from utils import UNK_token, SOS_token, EOS_token, asMinutes, timeSince, showPlot


###############################################
# Training settings
###############################################

parser = argparse.ArgumentParser(description='Sentence Reconstruction with Encoder-Decoder')
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data_path', type=str, default='/scratch/zc807/EncDec', metavar='PATH',
                    help='data path of pairs.pkl and lang.pkl (default: /scratch/zc807/EncDec)')
parser.add_argument('--save_data_path', type=str, default='/scratch/zc807/nlu/EncDec', metavar='PATH',
                    help='data path to save model parameters (default: /scratch/zc807/EncDec)')
parser.add_argument('--metric', type=str, default='MULTI', metavar='METRIC',
                    help='metric to use (default: MULTI; ROUGE, BLEU and BLEU_clip available)')
parser.add_argument('--vocab_size', type=int, default='15000', metavar='N',
                    help='vocab size (default: 15000)')
parser.add_argument('--hidden_size', type=int, default='300', metavar='N',
                    help='hidden size (default: 300)')
parser.add_argument('--n_epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--n_iters', type=int, default=3000, metavar='N',
                    help='number of iters to train (default: 3000), for testing only')
parser.add_argument('--print_every', type=int, default='1000', metavar='N',
                    help='print every (default: 1000) iters')
parser.add_argument('--plot_every', type=int, default='100', metavar='N',
                    help='plot every (default: 100) iters')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--clip', type=float, default=10, metavar='CLIP',
                    help='gradient clip threshold (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.set_defaults(max_length=30)


###############################################
# Preparing training data
###############################################

class Lang:
    def __init__(self, lang_load):
        self.word2index = lang_load[0]
        self.word2count = lang_load[1]
        self.index2word = lang_load[2]
        self.n_words = lang_load[3]

def indexesFromSentence(lang, sentence, args):
    result = []
    for word in sentence.split(' '):
        if word in lang.word2index and lang.word2index[word] < args.vocab_size:
            result.append(lang.word2index[word])
        else:
            result.append(UNK_token)
    return result

def variableFromSentence(lang, sentence, args):
    use_cuda = args.cuda
    indexes = indexesFromSentence(lang, sentence, args)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes).view(-1, 1)
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair, input_lang, output_lang, args):
    input_variable = variableFromSentence(input_lang, pair[0], args)
    target_variable = variableFromSentence(output_lang, pair[1], args)
    return (input_variable, target_variable)


###############################################
# Training
###############################################

teacher_forcing_ratio = 0.5
 
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, args):
    use_cuda = args.cuda
    max_length = args.max_length

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = min(args.max_length, input_variable.size()[0])
    target_length = target_variable.size()[0]

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.item()

            decoder_input = torch.LongTensor([[ni]])
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    # Clip gradient
    nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
    nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainEpochs(encoder, decoder, input_lang, output_lang, pairs, args):
    n_epochs = args.n_epochs
    print_every = args.print_every
    plot_every = args.plot_every
    learning_rate = args.lr

    start = time.time()
    iter_i = 0
    n_iters = n_epochs * len(pairs)
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(args.n_epochs):
        random.shuffle(pairs)
        training_pairs = [variablesFromPair(pair, input_lang, output_lang, args)
                      for pair in pairs]

        for training_pair in training_pairs:
            iter_i += 1
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, args)
            print_loss_total += loss
            plot_loss_total += loss

            if iter_i % args.print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter_i / n_iters),
                                            iter_i, iter_i / n_iters * 100, print_loss_avg))

            if iter_i % args.plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        
        print("Epoch {}/{} finished".format(epoch, args.n_epochs - 1))

    showPlot(plot_losses, args)


###############################################
# Evaluation
###############################################

def evaluate(encoder, decoder, sentence, input_lang, output_lang, args):
    use_cuda = args.cuda
    max_length = args.max_length

    input_variable = variableFromSentence(input_lang, sentence, args)
    input_length = min(args.max_length, input_variable.size()[0])
    encoder_hidden = encoder.initHidden()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = torch.LongTensor([[SOS_token]])  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
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
        pair = random.choice(pairs)
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

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.hpc:
        args.data_path = '.'
        args.save_data_path = '.'

    # Print settings
    print("hpc mode: {}".format(args.hpc))
    print("metric: {}".format(args.metric))
    print("vocab-size: {}".format(args.vocab_size))
    print("hidden-size: {}".format(args.hidden_size))
    print("n-epochs: {}".format(args.n_epochs))
    print("print-every: {}".format(args.print_every))
    print("plot-every: {}".format(args.plot_every))
    print("lr: {}".format(args.lr))
    print("clip: {}".format(args.clip))
    print("use cuda: {}".format(args.cuda))

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load pairs.pkl and lang.pkl
    with open(args.data_path + "/pairs.pkl", 'rb') as f:
        (train_pairs, test_pairs) = pkl.load(f)
    with open(args.data_path + "/lang.pkl", 'rb') as f:
        lang_tuple = pkl.load(f)
    lang = Lang(lang_tuple)

    # Set encoder and decoder
    encoder = EncoderRNN(args.vocab_size, args.hidden_size, args.cuda)
    decoder = DecoderRNN(args.hidden_size, args.vocab_size, args.cuda)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # Train and evalute
    print("\nStart")
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, decoder, train_pairs, lang, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, decoder, test_pairs, lang, lang, args)
    trainEpochs(encoder, decoder, lang, lang, train_pairs, args)
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, decoder, train_pairs, lang, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, decoder, test_pairs, lang, lang, args)
    evaluateTestingPairs(encoder, decoder, test_pairs, lang, lang, args)
    print("Finished\n")

    # Export trained weights
    torch.save(encoder.state_dict(), args.save_data_path + "/encoder_state_dict.pt")
    torch.save(decoder.state_dict(), args.save_data_path + "/decoder_state_dict.pt")
