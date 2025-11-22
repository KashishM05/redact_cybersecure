#!/usr/bin/env python3
"""
severity.py
Usage:
python severity.py --csv "C:\path\to\train.csv" --out "C:\path\to\severity_output.csv" --shap_plot "C:\path\to\shap.png"
This version expects Attack_encode with codes:
0 -> Benign
1 -> BruteForce
2 -> DoS
3 -> Malware
4 -> Scan
5 -> WebAttack
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def safe_float_series(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

parser = argparse.ArgumentParser()
parser.add_argument("--csv", "-c", required=True, help="Path to dataset CSV")
parser.add_argument("--out", "-o", default="severity_output.csv", help="Output CSV path")
parser.add_argument("--shap_plot", default="shap_summary.png", help="SHAP summary plot path")
args = parser.parse_args()

if not os.path.exists(args.csv):
    print("CSV not found:", args.csv)
    sys.exit(1)

df = pd.read_csv(args.csv)

# ---------------------------
# Features (use exact column names you provided)
# ---------------------------
feature_cols = [
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "URG Flag Count",
    "Flow IAT Mean", "Flow IAT Std", "Active Max", "Idle Max",
    "Packet Length Std", "Fwd Packet Length Std", "Bwd Packet Length Std",
    "Init Fwd Win Bytes"
]

present_features = [f for f in feature_cols if f in df.columns]
if len(present_features) < 5:
    print("Too few expected feature columns were found. Found:", present_features)
    sys.exit(1)

X_df = df[present_features].apply(safe_float_series)
scaler = MinMaxScaler()
X = scaler.fit_transform(X_df)

# ---------------------------
# Map Attack_encode -> textual label (hardcoded mapping)
# ---------------------------
mapping = {
    0: "Benign",
    1: "BruteForce",
    2: "DoS",
    3: "Malware",
    4: "Scan",
    5: "WebAttack"
}

if "Attack_encode" in df.columns:
    try:
        # if values are floats/strings, coerce to int where possible
        codes = pd.to_numeric(df["Attack_encode"], errors="coerce").fillna(-1).astype(int)
        mapped = codes.map(mapping).fillna("Unknown").astype(str)
        df["mapped_label"] = mapped
    except Exception:
        # fallback: convert each value safely
        def safe_map(v):
            try:
                k = int(v)
                return mapping.get(k, "Unknown")
            except:
                return "Unknown"
        df["mapped_label"] = df["Attack_encode"].apply(safe_map)
else:
    # If Attack_encode missing but Attack_type present, use Attack_type
    if "Attack_type" in df.columns:
        df["mapped_label"] = df["Attack_type"].astype(str)
    else:
        print("Neither Attack_encode nor Attack_type found. One of them is required.")
        sys.exit(1)

label_series = df["mapped_label"].astype(str)

# ---------------------------
# Create binary label for model (used only for SHAP importance training)
# ---------------------------
y_binary = (label_series.str.lower() != "benign").astype(int).values
if len(np.unique(y_binary)) < 2:
    print("Label column contains only one class after mapping; need at least two classes.")
    sys.exit(1)

import joblib

# ---------------------------
# Load Random Forest Model
# ---------------------------
try:
    # Try loading from model directory (relative to root)
    model_path = "model/random_forest.pkl"
    if not os.path.exists(model_path):
         # Fallback: try relative to script if run from elsewhere
         model_path = os.path.join(os.path.dirname(__file__), "../model/random_forest.pkl")
    
    model = joblib.load(model_path)
    print(f"Loaded {model_path} successfully.")
except Exception as e:
    print(f"Error loading random_forest.pkl: {e}")
    sys.exit(1)



# ---------------------------
# Label boost values tuned for your encoded labels
# (these are sensible defaults; you can tune later)
# ---------------------------
label_boost_map = {
    "benign": -2.0,
    "bruteforce": 0.9,
    "dos": 1.2,
    "malware": 1.3,
    "scan": 0.7,
    "webattack": 1.0,
    "unknown": 0.5
}

def compute_label_boost_from_mapped(lbl):
    l = str(lbl).lower()
    for k, v in label_boost_map.items():
        if k in l:
            return v
    return 0.5

label_boosts = label_series.apply(compute_label_boost_from_mapped).values
label_boosts_scaled = label_boosts / 2.0

# ---------------------------
# Compute severity (0..1)
# ---------------------------
# Since SHAP is removed, we'll use a simplified weight heuristic or uniform weights
# For now, let's assume uniform weights if we can't compute them dynamically without SHAP
# Or better, let's just use the label boost as the primary driver if feature weights aren't available
# However, to keep the structure, let's assign equal weights to all features.
weights = np.ones(X.shape[1]) / X.shape[1]

raw_feature_score = X.dot(weights)
severity_raw = raw_feature_score + label_boosts_scaled
severity = sigmoid(severity_raw)

df["severity_raw"] = severity_raw
df["severity"] = severity

# ---------------------------
# Generate Unique IDs
# ---------------------------
df["generated_id"] = range(1, len(df) + 1)

# Save output CSV
out_path = args.out
# Ensure generated_id is the first column for better readability
cols = ["generated_id"] + [c for c in df.columns if c != "generated_id"]
df[cols].to_csv(out_path, index=False)
print("Saved severity results to:", out_path)
