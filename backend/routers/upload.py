from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
import pandas as pd
import io
from routers.predict import model, predict_with_model, ATTACK_MAP
import numpy as np

router = APIRouter()

@router.post("/upload")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and get analysis similar to dashboard stats
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Limit to 100,000 rows for performance
        if len(df) > 100000:
            df = df.head(100000)
        
        # Validate required columns
        required_cols = ['Protocol', 'Total Fwd Packets', 'Total Backward Packets']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV is missing required columns: {', '.join(missing_cols)}"
            )
        
        # Filter out non-feature columns
        feature_cols = [col for col in df.columns if col not in ['Attack_type', 'Attack_encode']]
        X = df[feature_cols]
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Predict
        if model:
            preds = predict_with_model(model, X)
            pred_labels = [int(p) if isinstance(p, (int, float, np.number)) else int(p) for p in preds]
            pred_names = [ATTACK_MAP.get(p, 'Unknown') for p in pred_labels]
        else:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Calculate statistics
        total_flows = len(df)
        
        # Attack Distribution
        attack_counts = {}
        for name in pred_names:
            attack_counts[name] = attack_counts.get(name, 0) + 1
        
        # Protocol Distribution (All)
        protocol_counts = {}
        if 'Protocol' in df.columns:
            protocol_counts = df['Protocol'].value_counts().head(10).to_dict()
        
        # Protocol Distribution (Malicious)
        malicious_protocol_counts = {}
        recent_threats = []
        
        # Create temporary dataframe with predictions
        temp_df = df.copy()
        temp_df['Predicted_Attack'] = pred_names
        
        malicious_df = temp_df[temp_df['Predicted_Attack'] != 'Benign']
        
        if not malicious_df.empty:
            if 'Protocol' in malicious_df.columns:
                malicious_protocol_counts = malicious_df['Protocol'].value_counts().head(10).to_dict()
            
            # Recent Threats (last 20)
            threats_df = malicious_df.tail(20).iloc[::-1]
            
            for idx, row in threats_df.iterrows():
                recent_threats.append({
                    "id": int(idx),
                    "attack": row['Predicted_Attack'],
                    "protocol": str(row['Protocol']) if 'Protocol' in row else "Unknown",
                    "severity": "High",
                    "fwd_packets": int(row.get('Total Fwd Packets', 0)),
                    "bwd_packets": int(row.get('Total Backward Packets', 0))
                })
        
        return {
            "success": True,
            "filename": file.filename,
            "total_flows": total_flows,
            "attack_counts": attack_counts,
            "protocol_counts": protocol_counts,
            "malicious_protocol_counts": malicious_protocol_counts,
            "recent_threats": recent_threats
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/feature-importance")
async def calculate_feature_importance(file: UploadFile = File(...)):
    """
    Calculate feature importance (SHAP values) for uploaded CSV file
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Limit to 10,000 rows for SHAP calculation (performance)
        if len(df) > 10000:
            df = df.head(10000)
        
        # Filter out non-feature columns
        feature_cols = [col for col in df.columns if col not in ['Attack_type', 'Attack_encode']]
        X = df[feature_cols]
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Get feature importance from model
        if model:
            try:
                # For XGBoost models, get feature importance
                if hasattr(model, 'get_score'):
                    # XGBoost Booster
                    importance_dict = model.get_score(importance_type='weight')
                    
                    # Map feature names
                    feature_names = X.columns.tolist()
                    importances = {}
                    
                    for i, fname in enumerate(feature_names):
                        # XGBoost uses f0, f1, f2... as feature names
                        key = f'f{i}'
                        if key in importance_dict:
                            importances[fname] = float(importance_dict[key])
                        else:
                            importances[fname] = 0.0
                    
                    return {
                        "success": True,
                        "importances_dict": importances,
                        "feature_count": len(importances)
                    }
                elif hasattr(model, 'feature_importances_'):
                    # Sklearn-style model
                    feature_names = X.columns.tolist()
                    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
                    
                    return {
                        "success": True,
                        "importances_dict": importances,
                        "feature_count": len(importances)
                    }
                else:
                    raise HTTPException(status_code=500, detail="Model does not support feature importance")
                    
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Error calculating importance: {str(e)}")
        else:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
