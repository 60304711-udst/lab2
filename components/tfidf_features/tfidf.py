import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    parser = argparse.ArgumentParser()
    # This component takes all splits to prevent data leakage
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--deploy_data", type=str, required=True) # New deploy input
    
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True) # New deploy output
    
    # We limit max_features so the cloud computer doesn't run out of RAM
    parser.add_argument("--max_features", type=int, default=100)
    args = parser.parse_args()

    print("Loading data splits...")
    train_df = pd.read_parquet(args.train_data)
    val_df = pd.read_parquet(args.val_data)
    test_df = pd.read_parquet(args.test_data)
    deploy_df = pd.read_parquet(args.deploy_data) # Load deploy data

    text_col = 'reviewText'
    
    # Ensure no null values break the vectorizer
    train_df[text_col] = train_df[text_col].fillna("").astype(str)
    val_df[text_col] = val_df[text_col].fillna("").astype(str)
    test_df[text_col] = test_df[text_col].fillna("").astype(str)
    deploy_df[text_col] = deploy_df[text_col].fillna("").astype(str) # Handle nulls for deploy

    print(f"Fitting TF-IDF on training data only...")
    # Initialize TF-IDF with the professor's exact recommended settings
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )

    # CRITICAL: We FIT the model strictly on the training data!
    vectorizer.fit(train_df[text_col])

    def apply_tfidf(df, split_name):
        print(f"Transforming {split_name} data...")
        # .transform() applies the fitted model to the text without re-learning
        matrix = vectorizer.transform(df[text_col])
        
        # Convert the sparse matrix into a DataFrame with proper column names
        feature_names = [f"tfidf_{col}" for col in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(matrix.toarray(), columns=feature_names, index=df.index)
        
        # Merge the new TF-IDF features back with the original columns
        return pd.concat([df, tfidf_df], axis=1)

    # Apply the vectorizer to all splits
    train_df = apply_tfidf(train_df, "train")
    val_df = apply_tfidf(val_df, "validation")
    test_df = apply_tfidf(test_df, "test")
    deploy_df = apply_tfidf(deploy_df, "deployment") # Apply transform to deploy

    print("Writing outputs...")
    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.deploy_out, exist_ok=True) # Make directory for deploy

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"))
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"))
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"))
    deploy_df.to_parquet(os.path.join(args.deploy_out, "data.parquet")) # Save deploy data
    
    print("TF-IDF extraction complete!")

if __name__ == "__main__":
    main()