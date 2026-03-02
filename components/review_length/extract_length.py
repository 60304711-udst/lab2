import argparse
import os
import pandas as pd

def main():
    # Set up the inputs and outputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.data)

    text_column = 'reviewText'

    # Catch any null values just in case, treating them as empty strings
    df[text_column] = df[text_column].fillna("").astype(str)

    print("Calculating review lengths...")
    
    # Feature 1: Number of characters
    # We use .apply(len) to count every single character (including spaces) in the string
    df['review_length_chars'] = df[text_column].apply(len)

    # Feature 2: Number of words
    # We split the string by spaces (.split()) which creates a list of words, then count the length of that list
    df['review_length_words'] = df[text_column].apply(lambda x: len(x.split()))

    print("Writing outputs...")
    os.makedirs(args.out, exist_ok=True)
    
    # Save the updated dataframe (now containing our two new feature columns)
    df.to_parquet(os.path.join(args.out, "data.parquet"))
    print("Done!")

if __name__ == "__main__":
    main()