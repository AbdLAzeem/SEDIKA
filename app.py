import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import shap

# Global Plot Defaults
plt.rcParams['figure.facecolor'] = '#f8fafc'
PLOT_BG = "#e5e7eb" # Neutral light gray for high visibility
GRID_COLOR = "#9ca3af" # Distinct gridlines

# Styles
st.set_page_config(page_title="RT-IoT2022 IDS Dashboard", layout="wide")
st.markdown("""
<style>
    /* 1. White Background */
    .stApp {
        background-color: #ffffff !important;
    }

    /* 2. All Labels and Text as Red */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp th, .stApp td, .stMarkdown {
        color: #ff0000 !important; /* Pure Red */
    }

    .main-header {
        font-size: 2.8rem;
        color: #ff0000 !important;
        font-weight: 800;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* 3. Light Blue Buttons */
    div.stButton > button:first-child {
        background-color: #add8e6 !important; /* Light Blue */
        color: #ff0000 !important; /* Keeping text red as requested */
        border-radius: 8px;
        border: 1px solid #ff0000;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    
    div.stButton > button:hover {
        background-color: #87ceeb !important; /* Sky Blue on hover */
        color: #ffffff !important;
    }

    /* Professional cards for metrics (keeping white/red contrast) */
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 12px;
        border: 2px solid #ff0000;
    }
    
    /* Sidebar labels visibility */
    [data-testid="stSidebar"] * {
        color: #ff0000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Path Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "processed_data")

@st.cache_resource
def load_resources():
    # Supervised Models
    lgbm = joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
    dnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dnn.keras"))
    
    # Anomaly Models
    if_model = joblib.load(os.path.join(MODEL_DIR, "if_model.joblib"))
    ae_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "ae_model.keras"))
    ae_threshold = joblib.load(os.path.join(MODEL_DIR, "ae_threshold.joblib"))
    
    # Scaler & Encoder
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(DATA_DIR, "label_encoder.joblib"))
    
    # Test Data
    test_data = pd.read_pickle(os.path.join(DATA_DIR, "test_data.pkl"))
    
    return lgbm, dnn, if_model, ae_model, ae_threshold, scaler, le, test_data

try:
    lgbm_model, dnn_model, if_mod, ae_mod, ae_thresh, scaler, le, test_df = load_resources()
    st.sidebar.success("✅ Models & Scalers Loaded")
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Feature mapping
feature_cols = test_df.drop(columns=['target']).columns.tolist()

# App Header
st.markdown('<div class="main-header">🛡️ RT-IoT2022 Advanced IDS Dashboard</div>', unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["🚀 Real-time Monitor", "🔍 Threat Intelligence (SHAP)", "🧪 Anomaly Detection"])

# Shared State
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = test_df.sample(1).drop(columns=['target'])
    st.session_state['true_label'] = le.inverse_transform([test_df.loc[st.session_state['input_data'].index[0], 'target']])[0]

# --- TAB 1: MONITOR ---
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Traffic Control")
        if st.button("🎲 Generate Random Traffic"):
            sample = test_df.sample(1)
            st.session_state['input_data'] = sample.drop(columns=['target'])
            st.session_state['true_label'] = le.inverse_transform([sample['target'].values[0]])[0]
            st.rerun()
            
        model_choice = st.selectbox("Select Classification Engine", ["LightGBM (Fastest)", "DNN (Robust)"])
        
        # Simulated Stream Toggle
        run_sim = st.toggle("🛰️ Start Simulated Live Stream")
        
        # Display Current Data (Inverse Scaled for user readability)
        st.info(f"**Ground Truth:** {st.session_state['true_label']}")
        
        # Create unscaled version for display
        unscaled_df = pd.DataFrame(
            scaler.inverse_transform(st.session_state['input_data']), 
            columns=feature_cols,
            index=st.session_state['input_data'].index
        )
        display_df = unscaled_df.T
        display_df.columns = ["Captured Traffic Value"]
        st.dataframe(display_df, height=400)

    with col2:
        st.subheader("Prediction Analytics")
        
        # Real-time Loop Container
        placeholder = st.empty()
        
        if run_sim:
            import time as tm
            while run_sim:
                # Get new sample
                sample = test_df.sample(1)
                data = sample.drop(columns=['target']).values
                true_lbl = le.inverse_transform([sample['target'].values[0]])[0]
                
                # Predict
                if model_choice.startswith("LightGBM"):
                    pred_idx = lgbm_model.predict(data)[0]
                    conf = np.max(lgbm_model.predict_proba(data))
                else:
                    pred_probs = dnn_model.predict(data, verbose=0)
                    pred_idx = np.argmax(pred_probs, axis=1)[0]
                    conf = np.max(pred_probs)
                
                pred_label = le.inverse_transform([pred_idx])[0]
                
                with placeholder.container():
                    st.write(f"🔍 Analyzing packet: **{true_lbl}**")
                    c1, c2 = st.columns(2)
                    c1.metric("Predicted", pred_label)
                    c2.metric("Confidence", f"{conf*100:.1f}%")
                    
                    if "Thing_Speak" not in pred_label:
                        st.error(f"🚨 ATTACK DETECTED: {pred_label}")
                    else:
                        st.success("✅ TRAFFIC CLEAN")
                    
                    tm.sleep(1.5)
        
        elif st.button("🔍 Run Full Inspection"):
            data = st.session_state['input_data'].values
            
            # Predict
            if model_choice.startswith("LightGBM"):
                pred_idx = lgbm_model.predict(data)[0]
                conf = np.max(lgbm_model.predict_proba(data))
            else:
                pred_probs = dnn_model.predict(data, verbose=0)
                pred_idx = np.argmax(pred_probs, axis=1)[0]
                conf = np.max(pred_probs)
                
            pred_label = le.inverse_transform([pred_idx])[0]
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = conf * 100,
                title = {'text': f"Confidence: {pred_label}"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1e3a8a"},
                    'steps': [
                        {'range': [0, 50], 'color': "#fee2e2"},
                        {'range': [50, 80], 'color': "#fef3c7"},
                        {'range': [80, 100], 'color': "#d1fae5"}]
                }
            ))
            fig.update_layout(
                paper_bgcolor=PLOT_BG, 
                plot_bgcolor=PLOT_BG, 
                font=dict(color="#ff0000", size=14, family="Arial")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if "Thing_Speak" in pred_label:
                st.success("✅ SYSTEM SECURE: Normal Traffic Detected")
            else:
                st.error(f"🚨 ALERT: {pred_label.upper()} ATTACK DETECTED")

# --- TAB 2: SHAP ---
with tabs[1]:
    st.subheader("Explainable AI: Feature Contribution")
    st.markdown("Understanding *why* the model made a specific prediction.")
    
    if st.button("🧬 Explain Prediction"):
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(lgbm_model)
            shap_values = explainer.shap_values(st.session_state['input_data'])
            
            # Handle multi-class shap values
            # shap_values is a list of arrays (one per class)
            # Find the predicted class index
            if model_choice.startswith("LightGBM"):
                pred_idx = lgbm_model.predict(st.session_state['input_data'].values)[0]
            else:
                pred_idx = np.argmax(dnn_model.predict(st.session_state['input_data'].values, verbose=0), axis=1)[0]
            
            # Use SHAP for LightGBM (much faster for interactivity)
            curr_shap = shap_values[pred_idx][0] if isinstance(shap_values, list) else shap_values[0]
            
            # Feature Importance Bar Chart (Local) with REAL values
            real_vals = scaler.inverse_transform(st.session_state['input_data'])[0]
            real_vals_ser = pd.Series(real_vals, index=feature_cols)
            
            # Format labels: "Feature Name (Value: 123.4)"
            display_names = [f"{col} (Val: {real_vals_ser[col]:.2f})" for col in feature_cols]
            
            shap_df = pd.DataFrame({
                'Feature': display_names,
                'Contribution': curr_shap
            }).sort_values('Contribution', key=abs, ascending=False).head(10)
            
            fig = px.bar(shap_df, x='Contribution', y='Feature', orientation='h',
                         title="Impact Analysis (Feature & Real Unit Value)",
                         color='Contribution', color_continuous_scale='RdBu_r')
            
            # High-contrast Dark Background for SHAP specificity
            SHAP_THEME_BG = "#1e293b" # Dark slate/charcoal
            fig.update_layout(
                paper_bgcolor=SHAP_THEME_BG, 
                plot_bgcolor=SHAP_THEME_BG, 
                font=dict(color="#ff0000", size=13, family="Courier New, monospace"),
                xaxis=dict(gridcolor="#334155", zerolinecolor="#ef4444", zerolinewidth=2),
                yaxis=dict(gridcolor="#334155")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Detailed Insights:** Red bars indicate features pushing towards the current classification, blue bars push away.")

# --- TAB 3: ANOMALY ---
with tabs[2]:
    st.subheader("Zero-Day & Anomaly Detection")
    st.markdown("Independent check using **Unsupervised Learning** (Autoencoder & Isolation Forest).")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("#### Isolation Forest Analysis")
        score = -if_mod.score_samples(st.session_state['input_data'])[0]
        st.metric("Anomaly Score", f"{score:.4f}")
        if score > 0.5: # IF threshold heuristic
            st.warning("⚠️ High Anomaly Score: Potential Outlier")
        else:
            st.success("✅ Isolation Forest: Traffic patterns look typical.")
            
    with col_b:
        st.write("#### Autoencoder Reconstruction")
        data = st.session_state['input_data'].values
        reconstruction = ae_mod.predict(data, verbose=0)
        mse = np.mean(np.power(data - reconstruction, 2))
        
        st.metric("Reconstruction Error (MSE)", f"{mse:.6f}")
        st.progress(min(mse / (ae_thresh * 2), 1.0))
        
        if mse > ae_thresh:
            st.error(f"🚨 ANOMALY: MSE ({mse:.4f}) exceeded threshold ({ae_thresh:.4f})")
        else:
            st.success(f"✅ NORMAL: Traffic matches baseline patterns (MSE < {ae_thresh:.4f})")

    # Feature Profile Radar (Comparison)
    st.write("---")
    st.subheader("Traffic Signature vs Normal Baseline")
    
    # Calculate Normal Mean and Inverse Scale for Radar
    id_thing_speak = le.transform(['Thing_Speak'])[0]
    normal_scaled_mean = test_df[test_df['target'] == id_thing_speak].drop(columns=['target']).mean()
    
    # Inverse scale both for visualization
    current_real = scaler.inverse_transform(st.session_state['input_data'])[0]
    baseline_real = scaler.inverse_transform([normal_scaled_mean])[0]
    
    # Convert to series for easy indexing
    current_real_ser = pd.Series(current_real, index=feature_cols)
    baseline_real_ser = pd.Series(baseline_real, index=feature_cols)

    top_f = feature_cols[:8]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=current_real_ser[top_f].values,
        theta=top_f,
        name='Current Traffic (Real)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=baseline_real_ser[top_f].values,
        theta=top_f,
        fill='toself',
        name='Normal Baseline (Real)'
    ))
    
    # Auto-adjust range for real values
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor=GRID_COLOR),
            angularaxis=dict(gridcolor=GRID_COLOR),
            bgcolor=PLOT_BG
        ),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#ff0000", size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Project: IoT IDS RT-IoT2022")
st.sidebar.info("Dashboard v2.0 - With XAI & Anomaly Support")
