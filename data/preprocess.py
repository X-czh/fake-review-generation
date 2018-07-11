import pandas as pd


if __name__ == '__main__':
    # Read the two CSV files to pandas dataframes
    df_business = pd.read_csv('business.csv')
    df_review = pd.read_csv('review.csv')

    # Delete unneeded keys in df_business
    preserved_keys_business = ['business_id', 'name', 'city', 'state', 'categories']
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

    # Remove any duplicated reviews
    combo.drop_duplicates(inplace=True)

    # Remove any non-ASCII characters
    combo.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

    # Remove the new line characters in reviews
    combo.replace({r'\n+': ''}, regex=True, inplace=True)

    # Add space between alphanumerical characters and punctuation marks
    combo.replace({r'([a-zA-Z0-9])([,:;.!?"()])': r'\1 \2'}, regex=True, inplace=True)
    combo.replace({r'([,:;.!?"()])([a-zA-Z0-9])': r'\1 \2'}, regex=True, inplace=True)
    
    # Save to CSV
    save_file = combo.head(500000)
    save_file.to_csv('restaurant_review.csv', sep='\t', encoding='utf-8')
