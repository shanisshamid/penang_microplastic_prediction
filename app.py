import streamlit as st
import pandas as pd
import joblib
import base64

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Microplastic Concentration Predictor",
    layout="wide"
)

# ==================================================
# Background Image
# ==================================================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    rgba(0, 0, 0, 0.2),
                    rgba(0, 0, 0, 0.2)
                ),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("river_wallpaper.jpg")

# ==================================================
# Custom CSS Styling
# ==================================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Typography */
.big-title {
    font-size: 40px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.2em;
}

.subtitle {
    font-size: 16px;
    color: #e5e7eb;
}

/* HERO CARD */
.hero-card {
    padding: 2em 2.2em;
    border-radius: 20px;
    background: linear-gradient(
        rgba(15, 23, 42, 0.95),
        rgba(15, 23, 42, 0.88)
    );
    margin-bottom: 1.6em;
    box-shadow: 0 12px 28px rgba(0,0,0,0.45);
    text-align: center;
}

/* SOFT SECTIONS */
.soft-card {
    padding: 1.4em 1.8em;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.65);
    margin-bottom: 1.3em;
}

/* Inputs */
.stNumberInput > div > div > input {
    background-color: #020617 !important;
    color: #f8fafc !important;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.18);
    padding: 0.4em 0.6em;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #16a34a);
    color: white;
    font-weight: 600;
    border-radius: 16px;
    padding: 0.7em 1.8em;
    margin-top: 1.2em;
    border: none;
}

/* Result */
.result-card {
    padding: 1.9em;
    border-radius: 20px;
    background: linear-gradient(135deg, #2563eb, #16a34a);
    color: white;
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    margin-top: 1.8em;
}

/* Small text */
.caption {
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Header
# ==================================================
st.markdown("""
<div class="hero-card">
  <div class="big-title">🌊 Microplastic Concentration Predictor</div>
  <div class="subtitle">
    Penang Rivers • Gradient Boosting • Copula-based Monte Carlo Simulation • 2026
  </div>
 <div class="subtitle">
    <br>This application predicts <b>microplastic concentration (particles/L)</b>
    using physicochemical river water quality parameters.<br>
  </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# Load Model & Scaler
# ==================================================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("champion_gradientboost_model.pkl")
    scaler = joblib.load("scaler_aug.pkl")
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception:
    st.error("❌ Failed to load model or scaler. Please check deployment files.")
    st.stop()

# ==================================================
# Water Quality Inputs (Calm Working Area)
# ==================================================
st.markdown("""
<div class="soft-card">
<h4>🧪 Water Quality Inputs</h4>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", 10.0, 50.0, 28.0)
    ph = st.number_input("pH", 0.0, 14.0, 7.0)
    do = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 15.0, 6.5)

with col2:
    cdc = st.number_input("Conductivity (µS/cm)", 0.0, 50000.0, 500.0)
    turbidity = st.number_input("Turbidity (NTU)", 0.0, 2000.0, 10.0)

# ==================================================
# Prediction
# ==================================================
if st.button("🔍 Predict Microplastic Concentration"):

    X_input = pd.DataFrame(
        [[temperature, ph, do, cdc, turbidity]],
        columns=[
            "Temperature (°C)",
            "pH",
            "DO(mg/L)",
            "CDC(µs/cm)",
            "Turbidity(NTUs)"
        ]
    )

    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]

    st.markdown(
        f"""
        <div class="result-card">
            Predicted Microplastic Concentration<br>
            {prediction:.2f} particles / L
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# Disclaimer & Footer
# ==================================================
st.caption(
    "⚠️ Predictions are based on historical Penang river data and are intended "
    "for decision-support purposes only.")

st.markdown("---")
st.caption(
    "UNIVERSITI SAINS MALAYSIA • CDS590 Consultancy Project and Practicum • Predictive modelling under limited data availability"
)
