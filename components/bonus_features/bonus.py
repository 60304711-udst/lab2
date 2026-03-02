import argparse
import os
import pandas as pd

def calc_diversity(text):
    words = text.split()
    if len(words) == 0:
        return 0.0
    # Number of unique words divided by total words
    return len(set(words)) / len(words)

def calc_avg_word_len(text):
    words = text.split()
    if len(words) == 0:
        return 0.0
    # Total characters in all words divided by number of words
    return sum(len(word) for word in words) / len(words)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.data)
    text_col = 'reviewText'
    
    # Catch any nulls just in case
    df[text_col] = df[text_col].fillna("").astype(str)

    print("Calculating bonus features...")
    # Feature 1: Lexical Diversity
    df['lexical_diversity'] = df[text_col].apply(calc_diversity)
    
    # Feature 2: Average Word Length
    df['avg_word_length'] = df[text_col].apply(calc_avg_word_len)

    print("Writing outputs...")
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))
    print("Bonus features complete!")

if __name__ == "__main__":
    main()