import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.data)
    text_col = 'reviewText'

    # Catch any null values to prevent the model from crashing
    df[text_col] = df[text_col].fillna("").astype(str)

    print("Loading the Sentence Transformer model...")
    # 'all-MiniLM-L6-v2' is a lightweight, incredibly fast version of BERT.
    # It is perfect for cloud CPU environments so your lab doesn't take hours to run!
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Calculating semantic embeddings (this may take a few minutes)...")
    # .encode() turns a list of text into a matrix of dense numerical vectors
    embeddings = model.encode(df[text_col].tolist(), show_progress_bar=True)
    
    # Save the resulting vectors into the new column the professor requested
    df['bert_embedding'] = embeddings.tolist()

    print("Writing outputs...")
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))
    print("Done!")

if __name__ == "__main__":
    main()