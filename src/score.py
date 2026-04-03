import json
import os
import joblib
import numpy as np
import pandas as pd

model = None

def init():
    global model
    # Look for the model file recursively so we CANNOT miss it
    model_root = os.getenv("AZUREML_MODEL_DIR")
    model_path = None
    
    for root, dirs, files in os.walk(model_root):
        if "model.pkl" in files:
            model_path = os.path.join(root, "model.pkl")
            break

    if model_path is None:
        # Fallback to a direct path if search fails
        model_path = os.path.join(model_root, "model.pkl")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data_dict = json.loads(raw_data)
        df = pd.DataFrame(data_dict["data"])
        
        # All your original logic
        ignore_cols = ['asin', 'reviewerID', 'reviewText', 'summary', 'reviewerName', 
                       'reviewTime', 'unixReviewTime', 'overall', 'label']
        feature_cols = [c for c in df.columns if c not in ignore_cols and c != 'sbert_vector']
        
        X_numeric = df[feature_cols].select_dtypes(include=[np.number]).values
        
        if "sbert_vector" in df.columns:
            X_sbert = np.vstack(df['sbert_vector'].apply(np.array).values)
            X = np.hstack((X_numeric, X_sbert))
        else:
            X = X_numeric
            
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}