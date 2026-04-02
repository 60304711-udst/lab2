import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    # Updated ratios to match 60/15/15/10 assignment requirements
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--deploy_ratio", type=float, default=0.10)
    
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True) # New output argument
    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset
    df = pd.read_parquet(args.data)

    # 1. Sort by time for the Deployment split (most recent 10%)
    # This fulfills the assignment requirement to simulate data drift
    if 'review_year' in df.columns:
        df = df.sort_values(by='review_year', ascending=True)
    else:
        print("Warning: 'review_year' column missing. Splitting sequentially without explicit sorting.")

    # 2. Chronological split: Historical (90%) vs Deployment (10%)
    split_index = int(len(df) * (1 - args.deploy_ratio))
    historical_df = df.iloc[:split_index]
    deploy_df = df.iloc[split_index:]

    # 3. First random split on historical data: Train vs Temp (Val + Test)
    # Train needs to be 60% of the TOTAL data. Since historical_df is 90% of total,
    # train_size relative to historical_df is 0.60 / 0.90 = 2/3
    hist_train_ratio = args.train_ratio / (1 - args.deploy_ratio)
    
    train_df, temp_df = train_test_split(
        historical_df,
        train_size=hist_train_ratio,
        random_state=args.seed,
        shuffle=True
    )

    # 4. Second random split: Validation vs Test
    # Both are 15% of total data, so split temp_df 50/50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=args.seed,
        shuffle=True
    )

    # Write outputs
    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.deploy_out, exist_ok=True) # Create deploy output folder

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)
    deploy_df.to_parquet(os.path.join(args.deploy_out, "data.parquet"), index=False) # Save deploy data

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))
    print("Deployment rows:", len(deploy_df))

if __name__ == "__main__":
    main()