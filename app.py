"""
SEDIKA — Secure Edge Domain robust Intrusion Knowledge Architecture
Interactive Intrusion Detection Dashboard  |  RT-IoT2022 Benchmark
"""
import os, time
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from ml_utils import add_gaussian_noise
from artifacts import load_artifact

# ── Design tokens (validated palette — dataviz skill, July 2026) ─────────────
C_NAVY       = "#1e3a5f"   # header / sidebar accent
C_BLUE       = "#2a78d6"   # series-1 / primary CTA  (slot 1)
C_ORANGE     = "#eb6834"   # series-2               (slot 2)
C_AQUA       = "#1baf7a"   # series-3               (slot 3)
C_YELLOW     = "#eda100"   # series-4               (slot 4)
C_SURFACE    = "#fcfcfb"   # chart surface (light)
C_PAGE       = "#f8fafc"   # page plane
C_INK        = "#0b0b0b"   # primary ink
C_INK2       = "#52514e"   # secondary ink
C_MUTED      = "#898781"   # axis / grid labels
C_GRID       = "#e1e0d9"   # hairline gridlines
C_GOOD       = "#0ca30c"   # status: good
C_WARN       = "#fab219"   # status: warning
C_SERIOUS    = "#ec835a"   # status: serious
C_CRITICAL   = "#d03b3b"   # status: critical
C_BORDER     = "rgba(11,11,11,0.10)"

PLOTLY_BASE = dict(
    paper_bgcolor=C_SURFACE,
    plot_bgcolor=C_SURFACE,
    font=dict(color=C_INK, size=13, family="system-ui, -apple-system, Segoe UI, sans-serif"),
    margin=dict(l=16, r=16, t=40, b=16),
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEDIKA — IoT IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
/* ── Reset & base ── */
.stApp {{ background-color:{C_PAGE}; }}
section[data-testid="stSidebar"] {{ background-color:{C_NAVY}; }}
section[data-testid="stSidebar"] * {{ color:#ffffff !important; }}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {{ color:#c3d6f0 !important; }}

/* ── Header bar ── */
.sedika-header {{
  background: linear-gradient(135deg, {C_NAVY} 0%, #1c4f8a 100%);
  padding: 1.5rem 2rem 1.25rem;
  border-radius: 12px;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}}
.sedika-header h1 {{
  margin: 0; font-size: 1.9rem; font-weight: 700;
  color: #ffffff; letter-spacing: -0.02em;
}}
.sedika-header p {{
  margin: 0.2rem 0 0; font-size: 0.85rem; color: #a8c4e8;
}}

/* ── Metric cards ── */
.metric-card {{
  background: #ffffff;
  border: 1px solid {C_GRID};
  border-radius: 10px;
  padding: 1.1rem 1.25rem;
  text-align: center;
}}
.metric-card .val {{
  font-size: 2rem; font-weight: 700; color: {C_NAVY};
  font-variant-numeric: tabular-nums;
}}
.metric-card .lbl {{
  font-size: 0.8rem; color: {C_INK2}; margin-top: 0.15rem;
  text-transform: uppercase; letter-spacing: 0.05em;
}}

/* ── Status badges ── */
.badge {{
  display: inline-flex; align-items: center; gap: 6px;
  padding: 0.4rem 0.9rem; border-radius: 20px;
  font-size: 0.85rem; font-weight: 600;
}}
.badge-good     {{ background:#e8f8e8; color:#006300; border:1px solid #0ca30c; }}
.badge-warn     {{ background:#fff8e0; color:#7a5400; border:1px solid #eda100; }}
.badge-critical {{ background:#fce8e8; color:#8b1a1a; border:1px solid {C_CRITICAL}; }}

/* ── Section headings ── */
.section-title {{
  font-size: 1.05rem; font-weight: 600; color: {C_NAVY};
  border-left: 3px solid {C_BLUE}; padding-left: 0.6rem;
  margin-bottom: 0.8rem; margin-top: 0.4rem;
}}

/* ── Divider ── */
hr.sedika {{ border: none; border-top: 1px solid {C_GRID}; margin: 1.5rem 0; }}

/* ── Tabs ── */
button[data-baseweb="tab"] {{ font-weight: 500; color: {C_INK2}; }}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: {C_BLUE} !important; border-bottom-color: {C_BLUE} !important;
}}

/* ── Dataframe ── */
.dataframe {{ font-size: 0.82rem; }}
</style>
""", unsafe_allow_html=True)

# ── Paths — support deployed (demo_data/) and local (processed_data/) ─────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = (os.path.join(BASE_DIR, "processed_data")
             if os.path.isdir(os.path.join(BASE_DIR, "processed_data"))
             else os.path.join(BASE_DIR, "demo_data"))

# ── Resource loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_resources():
    import tensorflow as tf  # lazy — not needed until models are actually loaded
    lgbm     = joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl"))
    dnn      = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dnn.keras"))
    if_mod   = joblib.load(os.path.join(MODEL_DIR, "if_model.joblib"))
    ae_mod   = tf.keras.models.load_model(os.path.join(MODEL_DIR, "ae_model.keras"))

    # threshold — new path (sedika_ae_threshold.joblib) with legacy fallback
    thresh_candidates = [
        os.path.join(MODEL_DIR,  "sedika_ae_threshold.joblib"),
        os.path.join(DATA_DIR,   "ae_threshold.joblib"),
    ]
    ae_thresh_obj = next(
        (joblib.load(p) for p in thresh_candidates if os.path.exists(p)), None
    )
    # threshold object may be a plain float or a dict with 'threshold' key
    if isinstance(ae_thresh_obj, dict):
        ae_thresh = float(ae_thresh_obj.get("threshold", 1.0))
    elif ae_thresh_obj is not None:
        ae_thresh = float(ae_thresh_obj)
    else:
        ae_thresh = 1.0

    scaler   = load_artifact(os.path.join(DATA_DIR, "scaler.joblib"))
    le       = load_artifact(os.path.join(DATA_DIR, "label_encoder.joblib"))
    test_df  = pd.read_pickle(os.path.join(DATA_DIR, "test_data.pkl"))
    return lgbm, dnn, if_mod, ae_mod, ae_thresh, scaler, le, test_df

try:
    lgbm_model, dnn_model, if_mod, ae_mod, ae_thresh, scaler, le, test_df = load_resources()
except Exception as exc:
    st.error(f"**Resource load error:** {exc}")
    st.info("Run `python preprocess_data.py` then `python train_ml.py / train_dl.py / train_anomaly.py` first.")
    st.stop()

feature_cols = [c for c in test_df.columns if c != "target"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SEDIKA")
    st.markdown("**S**ecure **E**dge **D**omain robust  \n**I**ntrusion **K**nowledge **A**rchitecture")
    st.markdown("<hr class='sedika'/>", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Classification engine",
        ["DNN — Robust (recommended)", "LightGBM — Fast"],
        help="DNN maintains >97% accuracy under Gaussian noise (σ=0.1); LightGBM collapses to ~14%.",
    )
    use_dnn = model_choice.startswith("DNN")

    st.markdown("<hr class='sedika'/>", unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#7faed4'>Dataset: RT-IoT2022 · 10 models · 3 seeds  "
        "<br>Paper under review</small>",
        unsafe_allow_html=True,
    )

# ── Shared session state ──────────────────────────────────────────────────────
def _new_sample():
    s = test_df.sample(1)
    st.session_state["sample_x"] = s.drop(columns=["target"])
    st.session_state["sample_y"] = le.inverse_transform([int(s["target"].iloc[0])])[0]

if "sample_x" not in st.session_state:
    _new_sample()

sample_x: pd.DataFrame = st.session_state["sample_x"]
true_label: str        = st.session_state["sample_y"]

def _predict(data_2d):
    if use_dnn:
        probs = dnn_model.predict(data_2d, verbose=0)
        idx   = int(np.argmax(probs, axis=1)[0])
        conf  = float(np.max(probs))
    else:
        idx  = int(lgbm_model.predict(data_2d)[0])
        conf = float(np.max(lgbm_model.predict_proba(data_2d)))
    label = le.inverse_transform([idx])[0]
    return label, conf, idx

def _is_normal(label: str) -> bool:
    return "Thing_Speak" in label or "normal" in label.lower()

def _status_badge(label: str, conf: float) -> str:
    if _is_normal(label):
        return f'<span class="badge badge-good">✅ Normal — {label}</span>'
    if conf >= 0.85:
        return f'<span class="badge badge-critical">🚨 Attack — {label} ({conf*100:.0f}%)</span>'
    return f'<span class="badge badge-warn">⚠️ Suspicious — {label} ({conf*100:.0f}%)</span>'

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="sedika-header">'
    '<div>🛡️</div>'
    '<div>'
    '<h1>SEDIKA — IoT Intrusion Detection Dashboard</h1>'
    '<p>RT-IoT2022 &nbsp;·&nbsp; 10 models &nbsp;·&nbsp; SHAP explainability &nbsp;·&nbsp; '
    'Cross-domain transfer (DIFA) &nbsp;·&nbsp; Paper under review</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Headline metrics ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
for col, val, lbl in [
    (m1, "99.72%", "LightGBM accuracy"),
    (m2, "97.13%", "DNN at σ = 0.1 noise"),
    (m3, "77.96%", "Cross-domain transfer"),
    (m4, "0.51%",  "Anomaly false-positive rate"),
]:
    col.markdown(
        f'<div class="metric-card"><div class="val">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_monitor, tab_shap, tab_anomaly, tab_stress, tab_jitter = st.tabs([
    "📡 Real-Time Monitor",
    "🔍 SHAP Explainability",
    "🧪 Anomaly Detection",
    "📶 Robustness Stress-Test",
    "📉 Decision Cliff (Jitter)",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Real-Time Monitor
# ═══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    ctrl, vis = st.columns([1, 2], gap="large")

    with ctrl:
        st.markdown('<div class="section-title">Traffic Control</div>', unsafe_allow_html=True)

        if st.button("🎲 Sample Random Packet", use_container_width=True):
            _new_sample()
            sample_x  = st.session_state["sample_x"]
            true_label = st.session_state["sample_y"]
            st.rerun()

        st.markdown(f"**Ground truth:** `{true_label}`")
        st.markdown("<hr class='sedika'/>", unsafe_allow_html=True)

        # Live stream toggle
        run_sim = st.toggle("🛰️ Start Simulated Live Stream")

        st.markdown("<hr class='sedika'/>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Captured Traffic Features</div>', unsafe_allow_html=True)
        unscaled = pd.DataFrame(
            scaler.inverse_transform(sample_x),
            columns=feature_cols,
            index=sample_x.index,
        )
        st.dataframe(unscaled.T.rename(columns={unscaled.T.columns[0]: "Value"}), height=360)

    with vis:
        st.markdown('<div class="section-title">Prediction Analytics</div>', unsafe_allow_html=True)
        placeholder = st.empty()

        if run_sim:
            stream_log: list[dict] = []
            for _ in range(30):
                s = test_df.sample(1)
                x = s.drop(columns=["target"]).values
                y = le.inverse_transform([int(s["target"].iloc[0])])[0]
                pred, conf, _ = _predict(x)
                stream_log.append({"Packet": y, "Prediction": pred, "Confidence": f"{conf*100:.1f}%",
                                    "Status": "✅ Normal" if _is_normal(pred) else "🚨 Alert"})
                with placeholder.container():
                    st.markdown(_status_badge(pred, conf), unsafe_allow_html=True)
                    log_df = pd.DataFrame(stream_log[-10:][::-1])
                    st.dataframe(log_df, use_container_width=True, hide_index=True)
                time.sleep(1.2)
        else:
            if st.button("🔍 Classify This Packet", use_container_width=True):
                pred, conf, _ = _predict(sample_x.values)
                with placeholder.container():
                    st.markdown(_status_badge(pred, conf), unsafe_allow_html=True)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=conf * 100,
                        number={"suffix": "%", "font": {"color": C_NAVY}},
                        title={"text": f"Confidence — {pred}", "font": {"color": C_INK2, "size": 13}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": C_MUTED},
                            "bar": {"color": C_GOOD if _is_normal(pred) else C_CRITICAL, "thickness": 0.25},
                            "bgcolor": C_SURFACE,
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 50],  "color": "#fce8e8"},
                                {"range": [50, 80], "color": "#fff8e0"},
                                {"range": [80, 100],"color": "#e8f8e8"},
                            ],
                        },
                    ))
                    fig.update_layout(**PLOTLY_BASE, height=280)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Click **Classify This Packet** to run inference, or enable the live stream.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    st.markdown('<div class="section-title">SHAP Feature Attribution</div>', unsafe_allow_html=True)
    st.caption("Why did the model classify this packet? Top-10 features ranked by |SHAP|.")

    if st.button("🧬 Compute SHAP Values", use_container_width=False):
        with st.spinner("Computing SHAP values — this takes ~5 s on first call…"):
            explainer   = shap.TreeExplainer(lgbm_model)
            shap_values = explainer.shap_values(sample_x)
            pred, conf, pred_idx = _predict(sample_x.values)
            curr_shap = shap_values[pred_idx][0] if isinstance(shap_values, list) else shap_values[0]

            real_vals = scaler.inverse_transform(sample_x)[0]
            labels    = [f"{c}  ({real_vals[i]:.2f})" for i, c in enumerate(feature_cols)]

            shap_df = pd.DataFrame({"Feature": labels, "SHAP": curr_shap})\
                        .sort_values("SHAP", key=abs, ascending=False).head(10)

            # Diverging color: blue = helps class, red = hurts class
            shap_df["Color"] = shap_df["SHAP"].apply(
                lambda v: C_BLUE if v >= 0 else C_CRITICAL
            )

            fig = go.Figure(go.Bar(
                x=shap_df["SHAP"], y=shap_df["Feature"],
                orientation="h",
                marker_color=shap_df["Color"],
                marker_line_width=0,
            ))
            fig.update_layout(
                **PLOTLY_BASE,
                title=f"SHAP attribution — predicted: <b>{pred}</b> ({conf*100:.1f}%)",
                xaxis=dict(title="SHAP value (impact on prediction)", gridcolor=C_GRID,
                           zeroline=True, zerolinecolor=C_INK, zerolinewidth=1.5),
                yaxis=dict(gridcolor=C_GRID),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    '<span class="badge badge-good">🔵 Positive SHAP</span> &nbsp; '
                    'Feature pushes prediction <b>toward</b> this class',
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    '<span class="badge badge-critical">🔴 Negative SHAP</span> &nbsp; '
                    'Feature pushes prediction <b>away</b> from this class',
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown('<div class="section-title">Zero-Day Threat Detection</div>', unsafe_allow_html=True)
    st.caption("Tier-3 unsupervised detectors fire on traffic outside the training distribution.")

    col_if, col_ae = st.columns(2, gap="large")

    with col_if:
        st.markdown("**Isolation Forest**")
        score = float(-if_mod.score_samples(sample_x)[0])
        verdict = (
            (C_GOOD,     "badge-good",     "✅ Normal",   "Anomaly score within expected range")
            if score <= 0.5 else
            (C_CRITICAL, "badge-critical", "🚨 Anomaly",  "High anomaly score — potential zero-day")
        )
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="val" style="color:{verdict[0]}">{score:.4f}</div>'
            f'<div class="lbl">Anomaly Score</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<span class="badge {verdict[1]}">{verdict[2]}</span> — {verdict[3]}',
                    unsafe_allow_html=True)

    with col_ae:
        st.markdown("**Autoencoder Reconstruction (FPR-budget calibrated)**")
        recon = ae_mod.predict(sample_x.values, verbose=0)
        mse   = float(np.mean((sample_x.values - recon) ** 2))
        ratio = min(mse / (ae_thresh * 2), 1.0)

        if mse > ae_thresh:
            card_color = C_CRITICAL; bclass = "badge-critical"; btext = "🚨 Anomaly"
        else:
            card_color = C_GOOD;     bclass = "badge-good";     btext = "✅ Normal"

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="val" style="color:{card_color}">{mse:.6f}</div>'
            f'<div class="lbl">Reconstruction MSE  (threshold: {ae_thresh:.4f})</div></div>',
            unsafe_allow_html=True,
        )
        st.progress(ratio)
        st.markdown(f'<span class="badge {bclass}">{btext}</span>', unsafe_allow_html=True)

    # ── Radar: current traffic vs normal baseline ──
    st.markdown("<hr class='sedika'/>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Traffic Signature vs Normal Baseline</div>',
                unsafe_allow_html=True)

    id_normal  = le.transform(["Thing_Speak"])[0]
    normal_mean = test_df[test_df["target"] == id_normal].drop(columns=["target"]).mean()
    current_real = scaler.inverse_transform(sample_x)[0]
    baseline_real = scaler.inverse_transform([normal_mean])[0]

    top_f = feature_cols[:8]
    cur_r = pd.Series(current_real, index=feature_cols)[top_f]
    bas_r = pd.Series(baseline_real, index=feature_cols)[top_f]

    # normalise to [0,1] for radar readability
    combined_max = np.maximum(np.abs(cur_r.values), np.abs(bas_r.values))
    combined_max[combined_max == 0] = 1
    cur_n = np.abs(cur_r.values) / combined_max
    bas_n = np.abs(bas_r.values) / combined_max

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=cur_n, theta=top_f, name="Current packet",
                                  line=dict(color=C_ORANGE, width=2),
                                  fill="toself", fillcolor=f"rgba(235,104,52,0.12)"))
    fig.add_trace(go.Scatterpolar(r=bas_n, theta=top_f, name="Normal baseline",
                                  line=dict(color=C_BLUE, width=2),
                                  fill="toself", fillcolor=f"rgba(42,120,214,0.12)"))
    fig.update_layout(**PLOTLY_BASE, height=380,
                      polar=dict(radialaxis=dict(visible=True, gridcolor=C_GRID, range=[0,1]),
                                 angularaxis=dict(gridcolor=C_GRID),
                                 bgcolor=C_SURFACE))
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Robustness Stress-Test
# ═══════════════════════════════════════════════════════════════════════════════
with tab_stress:
    st.markdown('<div class="section-title">Wireless-Noise Robustness (Decision Cliff)</div>',
                unsafe_allow_html=True)
    st.caption(
        "Inject Gaussian noise to simulate wireless interference (IEEE 802.11 / 802.15.4). "
        "LightGBM collapses at σ = 0.1; SEDIKA's DNN maintains 97.13%."
    )

    noise_lvl = st.slider("Noise level σ", 0.0, 0.5, 0.05, 0.01)

    clean = sample_x.values
    noisy = add_gaussian_noise(clean, noise_level=noise_lvl)

    col_clean, col_noisy = st.columns(2, gap="large")
    with col_clean:
        st.markdown("**Clean signal**")
        pred_c, conf_c, _ = _predict(clean)
        st.markdown(_status_badge(pred_c, conf_c), unsafe_allow_html=True)
        st.metric("Confidence", f"{conf_c*100:.1f}%")

    with col_noisy:
        st.markdown(f"**Noisy signal (σ = {noise_lvl})**")
        pred_n, conf_n, _ = _predict(noisy)
        st.markdown(_status_badge(pred_n, conf_n), unsafe_allow_html=True)
        delta = f"{(conf_n - conf_c)*100:+.1f}%"
        st.metric("Confidence", f"{conf_n*100:.1f}%", delta=delta)

    if pred_c == pred_n:
        st.success("✅ Prediction stable under noise — model is robust at this σ.")
    else:
        st.error(f"⚠️ Prediction flipped: `{pred_c}` → `{pred_n}`. "
                 f"This is the Decision Cliff in action.")

    # ── Signal comparison chart ──
    st.markdown("<hr class='sedika'/>", unsafe_allow_html=True)
    feat_show = feature_cols[:12]
    comp_df = pd.DataFrame({
        "Feature": feat_show,
        "Clean":   clean[0][:12],
        "Noisy":   noisy[0][:12],
    }).set_index("Feature")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Clean",  x=feat_show, y=comp_df["Clean"],
                         marker_color=C_BLUE,   marker_line_width=0))
    fig.add_trace(go.Bar(name="Noisy",  x=feat_show, y=comp_df["Noisy"],
                         marker_color=C_ORANGE, marker_line_width=0, opacity=0.75))
    fig.update_layout(**PLOTLY_BASE, barmode="group", height=340,
                      xaxis=dict(gridcolor=C_GRID, title="Feature"),
                      yaxis=dict(gridcolor=C_GRID, title="Scaled value"),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP importance shift ──
    st.markdown('<div class="section-title">Feature Importance Shift under Noise</div>',
                unsafe_allow_html=True)
    if st.button("🧬 Compute SHAP Shift", use_container_width=False):
        with st.spinner("Computing…"):
            explainer  = shap.TreeExplainer(lgbm_model)
            _, _, pred_idx = _predict(clean)
            sv_clean = explainer.shap_values(clean)
            sv_noisy = explainer.shap_values(noisy)
            sc = sv_clean[pred_idx][0] if isinstance(sv_clean, list) else sv_clean[0]
            sn = sv_noisy[pred_idx][0] if isinstance(sv_noisy, list) else sv_noisy[0]

            shift_df = pd.DataFrame({"Feature": feature_cols,
                                     "Clean": np.abs(sc), "Noisy": np.abs(sn)})\
                         .sort_values("Clean", ascending=False).head(10)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Clean", x=shift_df["Feature"], y=shift_df["Clean"],
                                  marker_color=C_BLUE,   marker_line_width=0))
            fig2.add_trace(go.Bar(name="Noisy", x=shift_df["Feature"], y=shift_df["Noisy"],
                                  marker_color=C_CRITICAL, marker_line_width=0, opacity=0.8))
            fig2.update_layout(**PLOTLY_BASE, barmode="group", height=340,
                               xaxis=dict(gridcolor=C_GRID),
                               yaxis=dict(gridcolor=C_GRID, title="|SHAP|"),
                               legend=dict(orientation="h", y=1.08),
                               title="Top-10 |SHAP| — clean vs noisy input")
            st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Decision Cliff Jitter
# ═══════════════════════════════════════════════════════════════════════════════
with tab_jitter:
    st.markdown('<div class="section-title">Tree-Model Decision Cliff — Single-Feature Perturbation</div>',
                unsafe_allow_html=True)
    st.caption(
        "Vary one feature continuously. The vertical drop in probability reveals the "
        "orthogonal threshold that causes the Decision Cliff in tree-based models."
    )

    default_feat = "fwd_pkts_payload.avg" if "fwd_pkts_payload.avg" in feature_cols else feature_cols[0]
    jitter_feat  = st.selectbox("Feature to perturb", feature_cols,
                                index=feature_cols.index(default_feat))
    jitter_range = st.slider("Perturbation range (scaled units)", -2.0, 2.0,
                             (-0.5, 0.5), step=0.01)

    if st.button("📊 Run Cliff Analysis", use_container_width=False):
        with st.spinner("Scanning the decision boundary…"):
            sample_idx  = 100
            samp        = test_df.iloc[[sample_idx]].drop(columns=["target"]).copy()
            true_base   = le.inverse_transform([int(test_df.iloc[sample_idx]["target"])])[0]
            feat_idx    = feature_cols.index(jitter_feat)
            base_val    = float(samp.iloc[0, feat_idx])

            perturbs    = np.linspace(base_val + jitter_range[0],
                                     base_val + jitter_range[1], 300)
            perturbed   = pd.concat([samp] * 300).reset_index(drop=True)
            perturbed[jitter_feat] = perturbs

            probs       = lgbm_model.predict_proba(perturbed)
            preds       = np.argmax(probs, axis=1)
            base_cls    = preds[len(preds) // 2]
            prob_curve  = probs[:, base_cls]

            explainer   = shap.TreeExplainer(lgbm_model)
            sv           = explainer.shap_values(perturbed)
            shap_curve   = (sv[base_cls][:, feat_idx]
                            if isinstance(sv, list) else sv[:, feat_idx])

            # Two separate charts (no dual-axis — dataviz rule)
            fig_prob = go.Figure(go.Scatter(
                x=perturbs, y=prob_curve, mode="lines",
                line=dict(color=C_BLUE, width=2),
                fill="toself", fillcolor=f"rgba(42,120,214,0.08)",
                name="P(predicted class)",
            ))
            fig_prob.update_layout(
                **PLOTLY_BASE, height=260,
                title=f"Class probability as '{jitter_feat}' varies",
                xaxis=dict(title="Feature value (scaled)", gridcolor=C_GRID),
                yaxis=dict(title="Probability", range=[0, 1.05], gridcolor=C_GRID),
                showlegend=False,
            )

            fig_shap = go.Figure(go.Scatter(
                x=perturbs, y=shap_curve, mode="lines",
                line=dict(color=C_ORANGE, width=2),
                name="SHAP contribution",
            ))
            # Mark the cliff (max |dy/dx|)
            dy = np.gradient(shap_curve)
            cliff_idx = int(np.argmax(np.abs(dy)))
            fig_shap.add_vline(x=perturbs[cliff_idx],
                               line_dash="dash", line_color=C_CRITICAL, line_width=1.5,
                               annotation_text="Cliff", annotation_font_color=C_CRITICAL)
            fig_shap.update_layout(
                **PLOTLY_BASE, height=260,
                title=f"SHAP contribution — discontinuity = Decision Cliff",
                xaxis=dict(title="Feature value (scaled)", gridcolor=C_GRID),
                yaxis=dict(title="SHAP value", gridcolor=C_GRID),
                showlegend=False,
            )

            st.plotly_chart(fig_prob, use_container_width=True)
            st.plotly_chart(fig_shap, use_container_width=True)

            st.info(
                "**Reading this chart:** The vertical red dashed line marks where "
                f"the SHAP gradient is steepest — the orthogonal threshold of '{jitter_feat}'. "
                "A small feature shift here flips the prediction, demonstrating why "
                "LightGBM collapses to **13.85%** at σ = 0.1 while SEDIKA's DNN sustains **97.13%**."
            )
