import argparse
import os
import random
import pickle as pkl

import torch
import torch.backends.cudnn as cudnn

from model import EncoderBiRNN, DecoderRNN
from data import Lang, variableFromSentence
from utils import SOS_token, EOS_token

parser = argparse.ArgumentParser(description='seq2seq')
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data_path', type=str, default='.', metavar='PATH',
                    help='data path of pairs.pkl and lang.pkl (default: /scratch/zc807/seq2seq)')
parser.add_argument('--save_data_path', type=str, default='.', metavar='PATH',
                    help='data path to save model parameters (default: /scratch/zc807/seq2seq)')
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
parser.add_argument('--temperature', type=float, default=0.5, metavar='TEMP',
                    help='temperature (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1997, metavar='S',
                    help='random seed (default: 1997)')
parser.set_defaults(max_length=40)

def evaluate(encoder, decoder, sentence, input_lang, output_lang, args):
    
    with torch.no_grad():
        input_tensor = variableFromSentence(input_lang, sentence, args.vocab_size, args.cuda)
        input_lengths = [input_tensor.size(0)]
        _, encoder_hidden = encoder(input_tensor, input_lengths)

        decoder_input = torch.LongTensor([[SOS_token]])  # SOS
        decoder_input = decoder_input.cuda() if args.cuda else decoder_input
        
        # Concatenate bidirectional encoder hidden as context vector for decoder
        decoder_hidden = torch.cat([
            encoder_hidden[0:encoder_hidden.size(0):2], 
            encoder_hidden[1:encoder_hidden.size(0):2]
            ], 2)

        decoded_words = []

        for di in range(args.max_length):
            decoder_output, decoder_hidden = decoder.forward_step(
                decoder_input, decoder_hidden)
            word_weights = decoder_output.squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1).item()
            if word_idx == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[word_idx])

            decoder_input = torch.LongTensor([[word_idx]])
            decoder_input = decoder_input.cuda() if args.cuda else decoder_input

    return decoded_words

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, args, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        for j in range(5):
            print('>', pair[0])
            print('=', pair[1])
            output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang, args)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
    m = input("Enter the number of tests:\n")
    for i in range(int(m)):
        star = input('Enter rating star: (5.0 / 4.0 / 3.0 / 2.0 / 1.0)\n')
        category = input('Enter the category of the restaurant: (for example: "restaurant vietnamese)\n').lower()
        sent = star + ' ' + category
        for j in range(5):
            print('>', sent)
            output_words = evaluate(encoder, decoder, sent, input_lang, output_lang, args)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

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

    # Set encoder and decoder
    encoder = EncoderBiRNN(args.vocab_size, args.hidden_size, args.n_layers, args.dropout)
    decoder = DecoderRNN(args.hidden_size * 2, args.vocab_size, args.n_layers, args.dropout, args.cuda)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # Load checkpoint
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Load pairs.pkl and lang.pkl
    with open(args.data_path + "/pairs.pkl", 'rb') as f:
        (train_pairs, test_pairs) = pkl.load(f)
    with open(args.data_path + "/lang.pkl", 'rb') as f:
        lang_tuple = pkl.load(f)
    lang = Lang(lang_tuple)

    # Evaluate
    evaluateRandomly(encoder, decoder, test_pairs, lang, lang, args)
