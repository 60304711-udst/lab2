import argparse
import os
import time
import numpy as np
import pandas as pd
import joblib
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
def load_data(path):
    # Azure passes the folder path. We saved our files as "data.parquet" inside them.
    file_path = os.path.join(path, "data.parquet")
    if not os.path.exists(file_path):
        # Fallback just in case Azure passes the direct file path
        if os.path.exists(path) and path.endswith(".parquet"):
            file_path = path
        else:
            raise FileNotFoundError(f"Path does not exist: {file_path} (did you even pass the right input?)")
    
    return pd.read_parquet(file_path)

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------
def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing. You had one job.")
    
    # Binary classification: 1 if rating is 4 or 5, else 0
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

# ---------------------------------------------------------
# Features
# ---------------------------------------------------------
def build_features(df):
    """
    Combines TF-IDF, Sentiment, Length, and SBERT features into a single matrix.
    """
    if "sbert_vector" not in df.columns:
        print("Warning: 'sbert_vector' column not found. Attempting to auto-extract numeric features...")

    # 1. Define columns that are metadata or labels (NOT features)
    ignore_cols = ['asin', 'reviewerID', 'reviewText', 'summary', 'reviewerName', 'reviewTime', 'unixReviewTime', 'overall', 'label']
    
    # 2. Grab all standard numeric features (TF-IDF, sentiment, length)
    feature_cols = [c for c in df.columns if c not in ignore_cols and c != 'sbert_vector']
    X_numeric = df[feature_cols].select_dtypes(include=[np.number]).values

    # 3. Handle SBERT embeddings if they are stored as arrays in a single column
    if "sbert_vector" in df.columns:
        X_sbert = np.vstack(df['sbert_vector'].values)
        X = np.hstack((X_numeric, X_sbert)) # Stick them together
    else:
        X = X_numeric # Fallback to just the numeric columns

    if len(X) == 0:
        raise RuntimeError("Feature matrix is empty. Impressive.")
        
    return X

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate(model, X, y, split):
    # Get standard predictions (0 or 1)
    preds = model.predict(X)
    
    # Get probability of class 1 (required to calculate AUC)
    probs = model.predict_proba(X)[:, 1]
    
    # Calculate all metrics requested by the professor
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
    # Log everything to Azure MLflow
    mlflow.log_metric(f"{split}_accuracy", acc)
    mlflow.log_metric(f"{split}_auc", auc)
    mlflow.log_metric(f"{split}_precision", prec)
    mlflow.log_metric(f"{split}_recall", rec)
    mlflow.log_metric(f"{split}_f1", f1)
    
    print(f"[{split.upper()}] Acc: {acc:.4f} | AUC: {auc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    args = parse_args()
    start_time = time.time()

    print("Loading data... (this should not be the hard part)")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)

    print("Creating labels...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)

    print("Building features...")
    X_train = build_features(train_df)
    y_train = train_df["label"].values

    X_val = build_features(val_df)
    y_val = val_df["label"].values

    X_test = build_features(test_df)
    y_test = test_df["label"].values

    if len(X_train) == 0:
        raise RuntimeError("Training data is empty. That's concerning.")

    print("Training model...")
    # OPTIMIZATION: max_iter=1000 stops timeout warnings. n_jobs=-1 uses all CPU cores for max speed!
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    
    # Dump the file locally
    joblib.dump(model, model_path)
    
    # Tell MLflow to track this file as our official model artifact
    mlflow.log_artifact(model_path)

    # Calculate and log the final runtime for the bonus points
    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    
    print(f"Done. Total runtime: {runtime:.2f} seconds.")

if __name__ == "__main__":
    main()