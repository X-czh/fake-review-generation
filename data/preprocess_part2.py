import pandas as pd


if __name__ == '__main__':

    # Read the two CSV files to pandas dataframes
    df_train = pd.read_csv('train.csv', delimiter='\t')
    df_val = pd.read_csv('val.csv', delimiter='\t')

    for i in range(len(df_val)):
        if i % 1000 == 0:
            print(i)
        s = df_val['categories'][i]
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = s.replace("'", '')
        s = s.replace(',', '')
        df_val.loc[i, 'categories'] = s

    for i in range(len(df_train)):
        if i % 1000 == 0:
            print(i)
        s = df_train['categories'][i]
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = s.replace("'", '')
        s = s.replace(',', '')
        df_train.loc[i, 'categories'] = s

    # Save to CSV
    df_train.to_csv('train.csv', sep='\t', encoding='utf-8')
    df_val.to_csv('val.csv', sep='\t', encoding='utf-8')
