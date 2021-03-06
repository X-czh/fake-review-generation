import pandas as pd


if __name__ == '__main__':

    # Read the two CSV files to pandas dataframes
    df_business = pd.read_csv('business.csv')
    df_review = pd.read_csv('review.csv')

    # Delete unneeded keys in df_business
    preserved_keys_business = ['business_id', 'name', 'categories']
    for key in df_business.keys():
        if key not in preserved_keys_business:
            del df_business[key]
    
    # Delete unneeded keys in df_review
    preserved_keys_review = ['text', 'business_id', 'stars']
    for key in df_review.keys():
        if key not in preserved_keys_review:
            del df_review[key]
    
    # Filter 'Restaurants' businesses
    df_restaurants = df_business[df_business['categories'].str.contains('Restaurants')]

    # Merge the reviews with restaurants by key 'business_id'
    combo = pd.merge(df_restaurants, df_review, how='outer', on='business_id')

    # Delete 'business_id' key as it is no longer needed after merging
    del combo['business_id']

    # Drop rows with at least one NaN value
    combo.dropna(inplace=True)

    # Remove reviews with more than 40 words (including punctuations) (downsample step 1)
    combo = combo[combo['text'].apply(lambda x: len(x.split(' ')) <= 40)]

    # Drop rows with at least one NaN value
    combo.dropna(inplace=True)

    # Remove any duplicated reviews
    combo.drop_duplicates(inplace=True)

    # Remove any non-ASCII characters
    combo.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

    # Lowercase all the letters
    combo['name'] = combo['name'].astype(str).str.lower()
    combo['categories'] = (combo['categories'].astype(str).str.lower()).astype(list)
    combo['text'] = combo['text'].astype(str).str.lower()

    # Remove the new line characters in reviews
    combo.replace({r'\n+': ''}, regex=True, inplace=True)

    # Add space between alphanumerical characters and punctuation marks
    combo.replace({r'([a-z0-9])([,:;.!?"()-])': r'\1 \2'}, regex=True, inplace=True)
    combo.replace({r'([,:;.!?"()-])([a-z0-9])': r'\1 \2'}, regex=True, inplace=True)

    # Remove reviews with more than 30 words (including punctuations) (downsample step 2)
    combo = combo[combo['text'].apply(lambda x: len(x.split(' ')) <= 30)]

    # Remove reviews with more than 5 words of categories info (including punctuations) (downsample step 3)
    df = df[df['categories'].apply(lambda x: len(str(x).split(' ')) <= 5)]

    # Remove special characters (downsample step 4)
    df = df[df['text'].str.contains(r'[`~@#\$%^\*=\+\[\]/<>]', regex=True)==False]
    
    # Drop rows with at least one NaN value
    combo.dropna(inplace=True)

    # Save to CSV
    combo.to_csv('restaurant_review.csv', sep='\t', encoding='utf-8')

    # Drop column 0
    df = pd.read_csv('restaurant_review.csv', delimiter='\t')
    df = df.drop(['Unnamed: 0'], axis=1)
    df.to_csv('restaurant_review.csv', sep='\t', encoding='utf-8')
