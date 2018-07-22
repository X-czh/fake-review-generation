import pandas as pd


if __name__ == '__main__':

    # Read the CSV file to pandas dataframe
    df = pd.read_csv('restaurant_review.csv', delimiter='\t')

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Drop rows with at least one NaN value
    df.dropna(inplace=True)

    # Split and save to CSV
    train_file = df[:200000].drop(['Unnamed: 0'], axis=1)
    train_file.to_csv('train.csv', sep='\t', encoding='utf-8')
    eval_file = df[200000:210000].drop(['Unnamed: 0'], axis=1)
    eval_file.to_csv('test.csv', sep='\t', encoding='utf-8')
