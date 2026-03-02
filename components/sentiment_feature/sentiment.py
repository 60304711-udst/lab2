import argparse
import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # NLTK needs us to download the specific 'vader' dictionary before using it
    nltk.download('vader_lexicon')

    print("Loading data...")
    df = pd.read_parquet(args.data)
    text_column = 'reviewText'

    # Catch any null values and treat them as empty strings
    df[text_column] = df[text_column].fillna("").astype(str)

    print("Calculating sentiment scores (this may take a minute)...")
    sia = SentimentIntensityAnalyzer()

    # Apply VADER to every review. It returns a dictionary like: 
    # {'neg': 0.1, 'neu': 0.5, 'pos': 0.4, 'compound': 0.8}
    # .apply(pd.Series) splits that dictionary into separate columns!
    sentiment_scores = df[text_column].apply(sia.polarity_scores).apply(pd.Series)

    # Rename the columns to exactly match the professor's rubric
    sentiment_scores = sentiment_scores.rename(columns={
        'pos': 'sentiment_pos',
        'neg': 'sentiment_neg',
        'neu': 'sentiment_neu',
        'compound': 'sentiment_compound'
    })

    # Join the new sentiment columns to our original dataframe
    df = pd.concat([df, sentiment_scores], axis=1)

    print("Writing outputs...")
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))
    print("Done!")

if __name__ == "__main__":
    main()