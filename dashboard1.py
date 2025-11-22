import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import xgboost as xgb
from groq import Groq
import base64
import json
import io

# Page config
st.set_page_config(
    page_title="Network IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .threat-high { background-color: #ff4444; color: white; padding: 5px 10px; border-radius: 5px; }
    .threat-medium { background-color: #ff8800; color: white; padding: 5px 10px; border-radius: 5px; }
    .threat-low { background-color: #ffbb33; color: white; padding: 5px 10px; border-radius: 5px; }
    .benign { background-color: #00C851; color: white; padding: 5px 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Security action mapping
SECURITY_ACTIONS = {
    'DoS': 'BLOCK IP + Rate Limiting',
    'BruteForce': 'BLOCK IP + Account Lockout',
    'Scan': 'LOG + Monitor Suspicious Activity',
    'Malware': 'QUARANTINE + Deep Scan',
    'WebAttack': 'BLOCK Request + WAF Rule Update',
    'Benign': 'Allow Traffic'
}

# Label boost values (from severity.py)
LABEL_BOOST_MAP = {
    'benign': -2.0,
    'bruteforce': 0.9,
    'dos': 1.2,
    'malware': 1.3,
    'scan': 0.7,
    'webattack': 1.0,
    'unknown': 0.5
}

def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def compute_label_boost(attack_label):
    """Get label boost value based on attack type"""
    label_lower = str(attack_label).lower()
    for key, value in LABEL_BOOST_MAP.items():
        if key in label_lower:
            return value
    return 0.5

def calculate_severity(flow_features, attack_label):
    """
    Calculate severity score using the same logic as severity.py
    
    Args:
        flow_features: pandas Series or dict of flow features
        attack_label: str, the predicted attack type
    
    Returns:
        float: severity score between 0 and 1
    """
    # Convert to numpy array properly
    if hasattr(flow_features, 'values'):
        # It's a pandas Series - use .values (attribute, not method)
        feature_values = flow_features.values
    elif isinstance(flow_features, dict):
        # It's a dictionary
        feature_values = np.array(list(flow_features.values()))
    else:
        # Already an array
        feature_values = np.array(flow_features)
    
    # Use uniform weights for features (simplified version)
    weights = np.ones(len(feature_values)) / len(feature_values)
    
    # Calculate feature-based score
    raw_feature_score = np.dot(feature_values, weights)
    
    # Get label boost
    label_boost = compute_label_boost(attack_label)
    label_boost_scaled = label_boost / 2.0
    
    # Combine and normalize with sigmoid
    severity_raw = raw_feature_score + label_boost_scaled
    severity = sigmoid(severity_raw)
    
    return severity

# Load model
@st.cache_resource
def load_model(model_name='xgb_classifier.pkl'):
    import joblib
    import os
    
    # Path to model file (relative to data directory where dashboard runs)
    # Check both ../model/ and ../data/ locations
    paths_to_check = [
        os.path.join('..', 'model', model_name),
        os.path.join('..', 'data', model_name),
        model_name
    ]
    
    model_path = None
    for p in paths_to_check:
        if os.path.exists(p):
            model_path = p
            break
            
    if not model_path:
        st.error(f"Model file '{model_name}' not found in ../model/, ../data/ or current directory.")
        st.stop()

    try:
        # 1. Try loading with joblib (standard for sklearn)
        return joblib.load(model_path)
    except:
        try:
            # 2. Try loading as XGBoost native
            model = xgb.Booster()
            model.load_model(model_path)
            return model
        except:
            try:
                # 3. Try standard pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"Failed to load model '{model_name}': {e}")
                st.stop()

def predict_with_model(model, data):
    """
    Wrapper to handle prediction for both XGBoost (DMatrix) and Sklearn (DataFrame) models.
    """
    # Check if it's an XGBoost Booster
    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(data)
        return model.predict(dmatrix)
    
    # Check if it's an XGBoost sklearn wrapper (XGBClassifier)
    # These can handle DataFrames directly, but sometimes need DMatrix if strictly native
    # Usually sklearn-API XGBoost models handle DataFrames fine.
    
    # Default to sklearn-style predict (works for RandomForest, XGBClassifier, etc.)
    try:
        return model.predict(data)
    except Exception as e:
        # Fallback: try converting to DMatrix if it failed and looks like XGBoost might be involved
        try:
            dmatrix = xgb.DMatrix(data)
            return model.predict(dmatrix)
        except:
            raise e

# Load test data

# Load test data
@st.cache_data
def load_data():
    df = pd.read_csv('test.csv')
    return df

def get_severity_color(severity):
    if severity >= 0.8:
        return 'threat-high'
    elif severity >= 0.5:
        return 'threat-medium'
    elif severity > 0:
        return 'threat-low'
    return 'benign'

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'threat_log' not in st.session_state:
    st.session_state.threat_log = []

# Sidebar
st.sidebar.title("🛡️ Network IDS Control")
page = st.sidebar.radio("Navigation", 
    ["📊 Dashboard", "🔴 Live Monitor", "📈 Analytics", "🎯 Manual Prediction"])

if st.sidebar.button("🗑️ Clear Cache & Reset"):
    st.cache_resource.clear()
    st.cache_data.clear()
    if 'predictions' in st.session_state:
        del st.session_state.predictions
    if 'model_id' in st.session_state:
        del st.session_state.model_id
    st.rerun()

model = load_model('random_forest.pkl')
df = load_data()

# Check if model changed and clear stale predictions
if 'model_id' not in st.session_state:
    st.session_state.model_id = id(model)
elif st.session_state.model_id != id(model):
    st.session_state.predictions = []
    st.session_state.model_id = id(model)
    st.cache_data.clear()  # Optional: clear other data caches if needed

# Prepare feature columns
feature_cols = [col for col in df.columns if col not in ['Attack_type', 'Attack_encode']]

# Report Generation Functions
def generate_threat_report_with_groq(attack_summary, classification_report, threat_statistics):
    """
    Generate comprehensive threat report using Groq API with Llama 8B model.
    Follows MITRE ATT&CK framework for credible recommendations.
    """
    try:
        # Initialize Groq client
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key:
            # Try environment variable as fallback
            import os
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            return None, "Groq API key not found. Please set GROQ_API_KEY in Streamlit secrets or environment variables."
        
        client = Groq(api_key=api_key)
        
        # Prepare prompt for Groq
        prompt = f"""You are a cybersecurity expert analyzing network intrusion detection system results. 
Generate a comprehensive threat analysis report following MITRE ATT&CK framework guidelines.

ATTACK SUMMARY:
{json.dumps(attack_summary, indent=2)}

CLASSIFICATION METRICS:
{json.dumps(classification_report, indent=2)}

THREAT STATISTICS:
{json.dumps(threat_statistics, indent=2)}

Generate a detailed report that includes:
1. Executive Summary
2. Attack Type Analysis (for each detected attack type)
3. MITRE ATT&CK Framework Mapping and Recommendations:
   - For each attack type, provide specific mitigation strategies based on MITRE ATT&CK techniques
   - Include prevention, detection, and response actions
   - Reference MITRE ATT&CK tactics and techniques
4. Risk Assessment
5. Recommended Actions (prioritized)
6. Prevention Strategies

Format the report professionally with clear sections. Be specific and actionable in recommendations.
Reference MITRE ATT&CK techniques and tactics where applicable.

Report:"""
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity expert specializing in network intrusion detection and MITRE ATT&CK framework. Provide detailed, actionable security recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=4000
        )
        
        report_text = chat_completion.choices[0].message.content
        return report_text, None
        
    except Exception as e:
        return None, f"Error generating report: {str(e)}"

def create_report_html(report_text, attack_summary, attack_counts, confusion_matrix_data=None):
    """Create an HTML report with visualizations"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Network IDS Threat Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .section {{
                background: white;
                padding: 25px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            h3 {{
                color: #764ba2;
            }}
            .report-content {{
                line-height: 1.8;
                white-space: pre-wrap;
            }}
            .attack-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
            }}
            .footer {{
                text-align: center;
                color: #666;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛡️ Network Intrusion Detection System</h1>
            <p>Comprehensive Threat Analysis Report</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Attack Distribution Summary</h2>
            <div class="attack-stats">
"""
    
    # Add attack statistics
    for attack_type, count in attack_counts.items():
        html_content += f"""
                <div class="stat-card">
                    <div class="stat-value">{count:,}</div>
                    <div>{attack_type}</div>
                </div>
"""
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>Detailed Threat Analysis</h2>
            <div class="report-content">
"""
    
    # Add the AI-generated report text
    html_content += report_text.replace('\n', '<br>')
    
    html_content += """
            </div>
        </div>
        
        <div class="footer">
            <p>This report was generated using AI-powered analysis following MITRE ATT&CK framework guidelines.</p>
            <p>For more information, visit: <a href="https://attack.mitre.org">MITRE ATT&CK</a></p>
        </div>
    </body>
    </html>
    """
    
    return html_content

# Main Dashboard
if page == "📊 Dashboard":
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("---")
    
    # Run batch prediction if not done
    if len(st.session_state.predictions) == 0:
        with st.spinner("Running initial predictions..."):
            X = df[feature_cols]
            predictions = predict_with_model(model, X)
            st.session_state.predictions = predictions
    
    predictions = st.session_state.predictions
    
    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_flows = len(predictions)
    benign_count = np.sum(predictions == 0)
    malicious_count = total_flows - benign_count
    detection_rate = (malicious_count / total_flows * 100) if total_flows > 0 else 0
    
    with col1:
        st.metric("Total Flows", f"{total_flows:,}")
    with col2:
        st.metric("Benign", f"{benign_count:,}", delta=f"{benign_count/total_flows*100:.1f}%")
    with col3:
        st.metric("Malicious", f"{malicious_count:,}", delta=f"{malicious_count/total_flows*100:.1f}%", delta_color="inverse")
    with col4:
        st.metric("Detection Rate", f"{detection_rate:.2f}%")
    with col5:
        if 'Attack_encode' in df.columns:
            y_true = df['Attack_encode'].values
            # Filter out NaN values
            valid_mask = ~np.isnan(y_true)
            if valid_mask.sum() > 0:
                from sklearn.metrics import recall_score
                recall = recall_score(y_true[valid_mask], predictions[valid_mask], average='weighted', zero_division=0)
                st.metric("Recall Score", f"{recall:.3f}")
            else:
                st.metric("Threats Logged", len(st.session_state.threat_log))
        else:
            st.metric("Threats Logged", len(st.session_state.threat_log))
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attack Distribution")
        attack_map = {0: 'Benign', 1: 'DoS', 2: 'BruteForce', 3: 'Scan', 4: 'Malware', 5: 'WebAttack'}
        pred_labels = [attack_map.get(p, 'Unknown') for p in predictions]
        attack_counts = pd.Series(pred_labels).value_counts()
        
        fig = px.pie(values=attack_counts.values, names=attack_counts.index,
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Attack Type Counts")
        fig = px.bar(x=attack_counts.index, y=attack_counts.values,
                     labels={'x': 'Attack Type', 'y': 'Count'},
                     color=attack_counts.index,
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Protocol Distribution (All Traffic)")
        protocol_counts = df['Protocol'].value_counts().head(10)
        fig = px.bar(x=protocol_counts.index, y=protocol_counts.values,
                     labels={'x': 'Protocol', 'y': 'Flow Count'},
                     color=protocol_counts.values,
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Protocol Distribution (Malicious Only)")
        malicious_df = df[predictions != 0]
        if len(malicious_df) > 0:
            mal_protocol = malicious_df['Protocol'].value_counts().head(10)
            fig = px.bar(x=mal_protocol.index, y=mal_protocol.values,
                         labels={'x': 'Protocol', 'y': 'Attack Count'},
                         color=mal_protocol.values,
                         color_continuous_scale='OrRd')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No malicious traffic detected")
    
    # Recent Detections
    st.markdown("---")
    st.subheader("🚨 Recent Threat Detections")
    
    malicious_indices = np.where(predictions != 0)[0]
    if len(malicious_indices) > 0:
        recent_threats = malicious_indices[-10:][::-1]
        
        threat_data = []
        for idx in recent_threats:
            attack_label = attack_map.get(predictions[idx], 'Unknown')
            flow_features = df.iloc[idx][feature_cols]
            severity = calculate_severity(flow_features, attack_label)
            threat_data.append({
                'Flow ID': idx,
                'Attack Type': attack_label,
                'Severity': f"{severity:.2f}",
                'Protocol': df.iloc[idx]['Protocol'],
                'Action': SECURITY_ACTIONS.get(attack_label, 'Monitor'),
                'Fwd Packets': int(df.iloc[idx]['Total Fwd Packets']),
                'Bwd Packets': int(df.iloc[idx]['Total Backward Packets'])
            })
        
        threat_df = pd.DataFrame(threat_data)
        st.dataframe(threat_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No threats detected in current dataset")

# Live Monitor
elif page == "🔴 Live Monitor":
    st.title("🔴 Real-Time Threat Monitor")
    st.markdown("Simulated real-time monitoring using test data")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("▶️ Start Monitoring" if not st.session_state.monitoring else "⏸️ Pause Monitoring"):
            st.session_state.monitoring = not st.session_state.monitoring
    
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.current_index = 0
            st.session_state.threat_log = []
            st.rerun()
    
    with col3:
        speed = st.slider("Speed", 1, 10, 2)
    
    # Monitoring display
    placeholder = st.empty()
    
    attack_map = {0: 'Benign', 1: 'BruteForce', 2: 'DoS', 3: 'Malware', 4: 'Scan', 5: 'WebAttack'}
    
    if st.session_state.monitoring:
        while st.session_state.monitoring and st.session_state.current_index < len(df):
            idx = st.session_state.current_index
            
            # Get prediction
            X_current = df.iloc[idx:idx+1][feature_cols]
            pred = predict_with_model(model, X_current)[0]
            attack_label = attack_map.get(pred, 'Unknown')
            flow_features = df.iloc[idx][feature_cols]
            severity = calculate_severity(flow_features, attack_label)
            
            # Log threats
            if pred != 0:
                st.session_state.threat_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'flow_id': idx,
                    'attack_type': attack_label,
                    'severity': severity,
                    'action': SECURITY_ACTIONS.get(attack_label, 'Monitor')
                })
            
            with placeholder.container():
                st.markdown(f"### Flow #{idx} - {datetime.now().strftime('%H:%M:%S')}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Attack Type", attack_label)
                col2.metric("Severity", f"{severity:.2f}")
                col3.metric("Protocol", df.iloc[idx]['Protocol'])
                col4.metric("Action", SECURITY_ACTIONS.get(attack_label, 'Monitor'))
                
                if pred != 0:
                    st.error(f"🚨 THREAT DETECTED: {attack_label}")
                    
                    st.markdown("**Key Flow Characteristics:**")
                    key_features = ['Total Fwd Packets', 'Total Backward Packets', 
                                  'Flow Bytes/s', 'SYN Flag Count', 'ACK Flag Count']
                    feature_vals = df.iloc[idx][key_features].to_dict()
                    
                    cols = st.columns(len(key_features))
                    for i, (feat, val) in enumerate(feature_vals.items()):
                        cols[i].metric(feat, f"{val:.2f}")
                else:
                    st.success("✅ Benign Traffic - Allowed")
                
                st.progress((idx + 1) / len(df))
                
                # Show recent threat log
                if len(st.session_state.threat_log) > 0:
                    st.markdown("---")
                    st.markdown("**Recent Threats:**")
                    log_df = pd.DataFrame(st.session_state.threat_log[-5:][::-1])
                    st.dataframe(log_df, use_container_width=True, hide_index=True)
            
            st.session_state.current_index += 1
            time.sleep(1.0 / speed)
            
            if st.session_state.current_index >= len(df):
                st.session_state.monitoring = False
                st.info("Monitoring complete - reached end of dataset")
                break

# Analytics
elif page == "📈 Analytics":
    st.title("📈 Detailed Analytics")
    
    if len(st.session_state.predictions) == 0:
        with st.spinner("Running predictions..."):
            X = df[feature_cols]
            predictions = predict_with_model(model, X)
            st.session_state.predictions = predictions
    
    predictions = st.session_state.predictions
    attack_map = {0: 'Benign', 1: 'DoS', 2: 'BruteForce', 3: 'Scan', 4: 'Malware', 5: 'WebAttack'}
    
    # Confusion Matrix
    if 'Attack_encode' in df.columns:
        y_true = df['Attack_encode'].values
        # Filter out NaN values
        valid_mask = ~np.isnan(y_true)
        
        if valid_mask.sum() > 0:
            y_true_clean = y_true[valid_mask]
            predictions_clean = predictions[valid_mask]
            
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix, classification_report
            
            cm = confusion_matrix(y_true_clean, predictions_clean)
            
            fig = px.imshow(cm, 
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=[attack_map.get(i, f'Class {i}') for i in range(len(cm))],
                            y=[attack_map.get(i, f'Class {i}') for i in range(len(cm))],
                            color_continuous_scale='Blues',
                            text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Threat Report Generation Section
            st.markdown("---")
            st.subheader("📄 Generate Comprehensive Threat Report")
            st.markdown("Generate a detailed threat analysis report with MITRE ATT&CK-based recommendations and visualizations.")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("The report will include attack classifications, MITRE ATT&CK framework recommendations, and actionable security measures.")
            
            with col2:
                if st.button("📥 Generate Report", type="primary", use_container_width=True):
                    with st.spinner("Generating comprehensive threat report using AI..."):
                        # Prepare attack summary
                        pred_labels = [attack_map.get(p, 'Unknown') for p in predictions]
                        attack_counts_dict = pd.Series(pred_labels).value_counts().to_dict()
                        
                        # Generate classification report
                        report = classification_report(y_true_clean, predictions_clean, output_dict=True)
                        
                        # Prepare classification report summary
                        report_summary = {}
                        for key, value in report.items():
                            if isinstance(value, dict):
                                report_summary[key] = {
                                    'precision': value.get('precision', 0),
                                    'recall': value.get('recall', 0),
                                    'f1-score': value.get('f1-score', 0),
                                    'support': value.get('support', 0)
                                }
                        
                        # Prepare threat statistics
                        malicious_count = np.sum(predictions != 0)
                        benign_count = np.sum(predictions == 0)
                        total_flows = len(predictions)
                        
                        threat_stats = {
                            'total_flows': int(total_flows),
                            'malicious_flows': int(malicious_count),
                            'benign_flows': int(benign_count),
                            'threat_percentage': float((malicious_count / total_flows * 100) if total_flows > 0 else 0),
                            'attack_distribution': attack_counts_dict
                        }
                        
                        # Generate report using Groq
                        report_text, error = generate_threat_report_with_groq(
                            attack_summary=attack_counts_dict,
                            classification_report=report_summary,
                            threat_statistics=threat_stats
                        )
                        
                        if error:
                            st.error(error)
                            st.info("💡 To use report generation, please set your GROQ_API_KEY in Streamlit secrets (st.secrets) or as an environment variable.")
                        else:
                            st.success("✅ Report generated successfully!")
                            
                            # Create HTML report with visualizations
                            attack_counts_series = pd.Series(pred_labels).value_counts()
                            html_report = create_report_html(
                                report_text=report_text,
                                attack_summary=attack_counts_dict,
                                attack_counts=attack_counts_dict,
                                confusion_matrix_data=None
                            )
                            
                            # Display preview
                            st.markdown("### 📋 Report Preview")
                            with st.expander("View Generated Report", expanded=True):
                                st.markdown(report_text)
                            
                            # Download button for HTML report
                            b64_html = base64.b64encode(html_report.encode()).decode()
                            href_html = f'<a href="data:text/html;base64,{b64_html}" download="threat_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html" style="background-color: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">📥 Download HTML Report</a>'
                            st.markdown(href_html, unsafe_allow_html=True)
                            
                            # Also provide text download
                            b64_txt = base64.b64encode(report_text.encode()).decode()
                            href_txt = f'<a href="data:text/plain;base64,{b64_txt}" download="threat_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt" style="background-color: #764ba2; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin-left: 10px;">📄 Download Text Report</a>'
                            st.markdown(href_txt, unsafe_allow_html=True)
        else:
            st.warning("No valid labels found in Attack_encode column")
    
    # Feature Importance
    st.markdown("---")
    st.subheader("Top 15 Important Features")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        fig = px.bar(x=[feature_cols[i] for i in indices], 
                     y=importances[indices],
                     labels={'x': 'Feature', 'y': 'Importance'},
                     color=importances[indices],
                     color_continuous_scale='Viridis')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation for malicious traffic
    st.markdown("---")
    st.subheader("Feature Correlation Heatmap (Malicious Traffic)")
    
    malicious_df = df[predictions != 0]
    if len(malicious_df) > 0:
        key_features = ['Total Fwd Packets', 'Total Backward Packets', 
                       'Flow Bytes/s', 'Flow Packets/s', 'Packet Length Mean',
                       'SYN Flag Count', 'ACK Flag Count', 'PSH Flag Count']
        key_features = [f for f in key_features if f in df.columns]
        
        corr = malicious_df[key_features].corr()
        
        fig = px.imshow(corr, 
                        labels=dict(color="Correlation"),
                        x=key_features, y=key_features,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        text_auto='.2f')
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary by attack type
    st.markdown("---")
    st.subheader("Statistical Summary by Attack Type")
    
    pred_labels = [attack_map.get(p, 'Unknown') for p in predictions]
    df_with_pred = df.copy()
    df_with_pred['Predicted_Attack'] = pred_labels
    
    selected_attack = st.selectbox("Select Attack Type", 
                                   sorted(df_with_pred['Predicted_Attack'].unique()))
    
    attack_subset = df_with_pred[df_with_pred['Predicted_Attack'] == selected_attack]
    
    summary_features = ['Total Fwd Packets', 'Total Backward Packets', 
                       'Flow Bytes/s', 'Packet Length Mean', 'Flow IAT Mean']
    summary_features = [f for f in summary_features if f in df.columns]
    
    st.dataframe(attack_subset[summary_features].describe(), use_container_width=True)

# Manual Prediction
elif page == "🎯 Manual Prediction":
    st.title("🎯 Manual Flow Classification")
    st.markdown("Upload a CSV or input features manually")
    
    tab1, tab2 = st.tabs(["📁 Upload CSV", "⌨️ Manual Input"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            test_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(test_df)} flows")
            
            if st.button("🔍 Classify Flows"):
                with st.spinner("Classifying..."):
                    X_test = test_df[feature_cols]
                    preds = predict_with_model(model, X_test)
                    
                    attack_map = {0: 'Benign', 1: 'DoS', 2: 'BruteForce', 
                                 3: 'Scan', 4: 'Malware', 5: 'WebAttack'}
                    
                    test_df['Prediction'] = [attack_map.get(p, 'Unknown') for p in preds]
                    # Calculate severity dynamically for each flow
                    severities = []
                    for i, p in enumerate(preds):
                        attack_label = attack_map.get(p, 'Unknown')
                        flow_features = test_df.iloc[i][feature_cols]
                        sev = calculate_severity(flow_features, attack_label)
                        severities.append(sev)
                    test_df['Severity'] = severities
                    test_df['Action'] = [SECURITY_ACTIONS.get(attack_map.get(p, 'Unknown'), 'Monitor') 
                                        for p in preds]
                    
                    st.dataframe(test_df[['Prediction', 'Severity', 'Action'] + feature_cols[:5]], 
                               use_container_width=True)
                    
                    csv = test_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, 
                                     "predictions.csv", "text/csv")
    
    with tab2:
        st.markdown("**Enter flow features:**")
        
        input_data = {}
        cols = st.columns(3)
        
        for i, feature in enumerate(feature_cols):
            col_idx = i % 3
            with cols[col_idx]:
                input_data[feature] = st.number_input(feature, value=0.0, 
                                                     format="%.2f", key=feature)
        
        if st.button("🔍 Classify Flow"):
            X_input = pd.DataFrame([input_data])
            pred = predict_with_model(model, X_input)[0]
            
            attack_map = {0: 'Benign', 1: 'DoS', 2: 'BruteForce', 
                         3: 'Scan', 4: 'Malware', 5: 'WebAttack'}
            attack_label = attack_map.get(pred, 'Unknown')
            # Calculate severity using flow features
            flow_features = pd.Series(input_data)
            severity = calculate_severity(flow_features, attack_label)
            action = SECURITY_ACTIONS.get(attack_label, 'Monitor')
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Attack Type", attack_label)
            col2.metric("Severity", f"{severity:.2f}")
            col3.metric("Recommended Action", action)
            
            if pred != 0:
                st.error(f"🚨 THREAT DETECTED: {attack_label}")
                st.markdown(f"**Recommended Action:** {action}")
            else:
                st.success("✅ Benign Traffic - Allow")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(f"Dashboard v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")