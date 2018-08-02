import pandas as pd


if __name__ == '__main__':

    # Read the CSV file to pandas dataframe
    df = pd.read_csv('restaurant_review.csv', delimiter='\t')

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Drop rows with at least one NaN value
    df.dropna(inplace=True)

    # Sample an equal number of reviews of different stars
    df_5star = df[df['stars'] == 5.0]
    df_4star = df[df['stars'] == 4.0]
    df_3star = df[df['stars'] == 3.0]
    df_2star = df[df['stars'] == 2.0]
    df_1star = df[df['stars'] == 1.0]
    df_train = pd.concat([df_5star[:5000], df_4star[:5000], 
        df_3star[:5000], df_2star[:5000], df_1star[:5000]])
    df_val = pd.concat([df_5star[5000:5500], df_4star[5000:5500], 
        df_3star[5000:5500], df_2star[5000:5500], df_1star[5000:5500]])
    df_test = pd.concat([df_5star[5500:6000], df_4star[5500:6000], 
        df_3star[5500:6000], df_2star[5500:6000], df_1star[5500:6000]])

    # Split and save to CSV
    train_file = df_train.drop(['Unnamed: 0'], axis=1)
    train_file.to_csv('train.csv', sep='\t', encoding='utf-8')
    val_file = df_val.drop(['Unnamed: 0'], axis=1)
    val_file.to_csv('val.csv', sep='\t', encoding='utf-8')
    test_file = df_test.drop(['Unnamed: 0'], axis=1)
    test_file.to_csv('test.csv', sep='\t', encoding='utf-8')
