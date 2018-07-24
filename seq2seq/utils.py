import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define constants
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

# Pad sequence
def pad_sequence(seq, max_seq_len):
    seq.extend([PAD_token for i in range(max_seq_len - len(seq))])
    return seq

# Extract N-Grams
def extract_ngrams(sent, order):
    ngrams = []

    # tokenization
    uwords = sent.split(' ')
    
    # extract ngrams
    for oo in range(1, order + 1):
        for ng in ([' '.join(t).strip() for t in zip(*[uwords[i:] for i in range(oo)])]):
            ngrams.append(ng)

    return ngrams

# Timeing and Plotting
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points, args):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(args.data_path + "/loss.pdf")
    plt.close(fig)
