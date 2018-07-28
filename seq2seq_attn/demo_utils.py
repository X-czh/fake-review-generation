import io
import random
import socket
import torch
import torchvision
import visdom
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data import variableFromSentence
from utils import PAD_token, UNK_token, SOS_token, EOS_token

vis = visdom.Visdom()
hostname = socket.gethostname()

def show_plot_visdom():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention ({})'.format(hostname)
    vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), win=attn_win, opts={'title': attn_win})

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    show_plot_visdom()
    plt.show()
    plt.close()

def evaluate_and_show_attention(encoder, decoder, sentence, input_lang, output_lang, 
        vocab_size, max_length, use_cuda):
    with torch.no_grad():
        input_tensor = variableFromSentence(input_lang, sentence, vocab_size, use_cuda)
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

        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di,:decoder_attention.size(2)] += \
                decoder_attention.squeeze(0).squeeze(0).cpu().data
            topv, topi = decoder_output.data.topk(1)
            ni = topi.item()
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[ni])

            decoder_input = torch.LongTensor([[ni]])
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, 
        vocab_size, max_length, use_cuda, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        input_sentence, target_sentence = pair
        print('>', input_sentence)
        print('=', target_sentence)
        output_words, attentions = evaluate_and_show_attention(
            encoder, decoder, input_sentence, input_lang, output_lang, vocab_size, max_length, use_cuda)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

        show_attention(input_sentence, output_words, attentions)
    
        # Show input, target, output text in visdom
        win = 'evaluted ({})'.format(hostname)
        text = '<p>&gt; {}</p><p>= {}</p><p>&lt; {}</p>'.format(input_sentence, target_sentence, output_sentence)
        vis.text(text, win=win, opts={'title': win})