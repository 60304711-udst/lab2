import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    # The 4 required feature inputs
    parser.add_argument("--length_data", type=str, required=True)
    parser.add_argument("--sentiment_data", type=str, required=True)
    parser.add_argument("--tfidf_data", type=str, required=True)
    parser.add_argument("--embedding_data", type=str, required=True)
    # The bonus feature input (set to not required just in case)
    parser.add_argument("--bonus_data", type=str, required=False)
    
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    print("Loading all feature datasets...")
    df_length = pd.read_parquet(args.length_data)
    df_sentiment = pd.read_parquet(args.sentiment_data)
    df_tfidf = pd.read_parquet(args.tfidf_data)
    df_embedding = pd.read_parquet(args.embedding_data)

    # We use the length dataframe as our base to start merging onto
    merged_df = df_length.copy()

    # Helper function to join on entity keys without duplicating original columns
    def merge_new_cols(base_df, new_df):
        # Keep the join keys + only the brand new feature columns
        cols_to_use = ['asin', 'reviewerID'] + [col for col in new_df.columns if col not in base_df.columns]
        # Perform the inner join on the entity keys as requested by the rubric
        return pd.merge(base_df, new_df[cols_to_use], on=['asin', 'reviewerID'], how='inner')

    print("Merging sentiment features...")
    merged_df = merge_new_cols(merged_df, df_sentiment)
    
    print("Merging TF-IDF features...")
    merged_df = merge_new_cols(merged_df, df_tfidf)
    
    print("Merging semantic embeddings...")
    merged_df = merge_new_cols(merged_df, df_embedding)

    if args.bonus_data:
        print("Merging bonus features...")
        try:
            df_bonus = pd.read_parquet(args.bonus_data)
            merged_df = merge_new_cols(merged_df, df_bonus)
        except Exception as e:
            print(f"Skipping bonus data: {e}")

    print("Writing final feature-enriched dataset...")
    os.makedirs(args.out, exist_ok=True)
    
    # Save the final dataset as a Parquet file
    merged_df.to_parquet(os.path.join(args.out, "data.parquet"))
    
    print(f"Merge complete! Final shape: {merged_df.shape}")

if __name__ == "__main__":
    main()