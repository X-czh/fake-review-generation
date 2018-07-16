import pandas as pd
import pickle as pkl


if __name__ == '__main__':

    # Read the CSV file to pandas dataframe
    df = pd.read_csv('gene.csv', delimiter='\t')

    # Read the vocab data
    with open('vocab.pkl', 'rb') as f:
        lang = pkl.load(f)

    # Write translated data
    with open('gene.data', 'w') as fout:
        for line in df['text']:
            word_list = [ ]
            for s in line.split(' '):
                index = int(s)
                if index == 0:
                    word_list.append('<EOS>') # end of sequence
                    break
                else:
                    word_list.append(lang.index2word(index))
            sent = ' '.join(word_list)
            fout.write(sent + '\n')
