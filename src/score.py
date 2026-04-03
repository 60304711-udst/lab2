import json
import os
import joblib
import numpy as np
import pandas as pd

model = None

def init():
    """
    This runs ONCE when the endpoint starts up. 
    It loads the model into memory so it's ready for requests.
    """
    global model
    
    # AZUREML_MODEL_DIR
    # pointing to where it downloaded your registered model.
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully!")

def run(raw_data):
    """
    This runs EVERY TIME the endpoint receives a request.
    """
    try:
        # 1. Parse the incoming JSON request
        data_dict = json.loads(raw_data)
        
        # 2. Convert the incoming data back into a pandas DataFrame
        df = pd.DataFrame(data_dict["data"])
        
        # 3. Apply feature logic from train.py
        ignore_cols = ['asin', 'reviewerID', 'reviewText', 'summary', 'reviewerName', 'reviewTime', 'unixReviewTime', 'overall', 'label']
        feature_cols = [c for c in df.columns if c not in ignore_cols and c != 'sbert_vector']
        
        X_numeric = df[feature_cols].select_dtypes(include=[np.number]).values
        
        if "sbert_vector" in df.columns:
            # Reconstruct the numpy arrays from the lists sent via JSON
            X_sbert = np.vstack(df['sbert_vector'].apply(np.array).values)
            X = np.hstack((X_numeric, X_sbert))
        else:
            X = X_numeric
            
        # Clean up any bad data just like in training
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X) == 0:
             return {"error": "Feature matrix is empty after processing."}

        # 4. Make the pridictions
        preds = model.predict(X)
        
        # 5. Return the predictions as a JSON response
        return {"predictions": preds.tolist()}

    except Exception as e:
        # return the error 
        return {"error": str(e)}