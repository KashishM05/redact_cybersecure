from fastapi import APIRouter
import pandas as pd

router = APIRouter()

import os

# Robust data path finding
DATA_PATH = "data/test.csv"
if not os.path.exists(DATA_PATH):
    # Try finding it relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    DATA_PATH = os.path.join(parent_dir, "data", "test.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    print(f"Warning: Test data not found at {DATA_PATH}")
    df = pd.DataFrame() # Empty dataframe to prevent crash on import

@router.get("/next/{index}")
def get_flow(index: int):
    try:
        if index >= len(df):
            return {"end": True}

        row = df.iloc[index]
        # Handle NaN values for JSON serialization
        # Use a simpler approach if where() fails
        d = row.to_dict()
        clean_d = {}
        for k, v in d.items():
            if pd.isna(v):
                clean_d[k] = None
            else:
                clean_d[k] = v
        
        return {
            "index": index,
            "flow": clean_d,
            "end": False
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc(), "end": True}

# Import necessary modules for prediction
from routers.predict import model, predict_with_model, ATTACK_MAP
import numpy as np

@router.get("/stats")
def get_dashboard_stats():
    try:
        if df.empty:
            return {"error": "No data loaded"}
            
        # Limit to 100,000 rows for stats calculation as requested
        limit = 100000
        stats_df = df.head(limit)
        
        # Perform Prediction on the loaded data (replicating dashboard1.py logic)
        # We need to predict to get the actual current model's view of the data
        # Filter out non-feature columns explicitly
        feature_cols = [col for col in stats_df.columns if col not in ['Attack_type', 'Attack_encode']]
        X = stats_df[feature_cols]
        
        # Predict
        if model:
            preds = predict_with_model(model, X)
            # Convert predictions to labels
            # preds might be floats from XGBoost, cast to int
            pred_labels = [int(p) if isinstance(p, (int, float, np.number)) else int(p) for p in preds]
            pred_names = [ATTACK_MAP.get(p, 'Unknown') for p in pred_labels]
        else:
            # Fallback if model not loaded (shouldn't happen if app started correctly)
            return {"error": "Model not loaded"}

        # 1. Total Flows
        total_flows = len(stats_df)
        
        # 2. Attack Distribution (based on PREDICTIONS)
        attack_counts = {}
        for name in pred_names:
            attack_counts[name] = attack_counts.get(name, 0) + 1
            
        # 3. Protocol Distribution (All)
        if 'Protocol' in stats_df.columns:
            protocol_counts = stats_df['Protocol'].value_counts().head(10).to_dict()
        else:
            protocol_counts = {}
            
        # 4. Protocol Distribution (Malicious)
        malicious_protocol_counts = {}
        recent_threats = []
        
        # Filter malicious based on predictions
        # Create a temporary dataframe with predictions for filtering
        temp_df = stats_df.copy()
        temp_df['Predicted_Attack'] = pred_names
        
        malicious_df = temp_df[temp_df['Predicted_Attack'] != 'Benign']
        
        if not malicious_df.empty:
            if 'Protocol' in malicious_df.columns:
                malicious_protocol_counts = malicious_df['Protocol'].value_counts().head(10).to_dict()
            
            # 5. Recent Threats (Take last 20 malicious flows)
            threats_df = malicious_df.tail(20).iloc[::-1]
            
            for idx, row in threats_df.iterrows():
                recent_threats.append({
                    "id": int(idx),
                    "attack": row['Predicted_Attack'],
                    "protocol": row['Protocol'] if 'Protocol' in row else "Unknown",
                    "severity": "High", # Simplified for summary
                    "fwd_packets": int(row.get('Total Fwd Packets', 0)),
                    "bwd_packets": int(row.get('Total Backward Packets', 0))
                })

        return {
            "total_flows": total_flows,
            "attack_counts": attack_counts,
            "protocol_counts": protocol_counts,
            "malicious_protocol_counts": malicious_protocol_counts,
            "recent_threats": recent_threats
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}
