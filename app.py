import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration (MUST be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="Microplastic Concentration Predictor",
    layout="wide"
)

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
st.title("🌊 Microplastic Concentration Predictor (Penang Rivers)")
st.markdown(
    """
    This application predicts **microplastic concentration (particles/L)**  
    using physicochemical river water quality parameters.

    **Model:** Gradient Boosting Regressor  
    **Approach:** Data augmentation using Copula-based Monte Carlo simulation
    """
)

# --------------------------------------------------
# Load Model & Scaler (Cached)
# --------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("champion_gradientboost_model.pkl")
    scaler = joblib.load("scaler_aug.pkl")
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error("❌ Failed to load model or scaler.")
    st.stop()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.subheader("🧪 Water Quality Inputs")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", value=28.0)
    ph = st.number_input("pH", value=7.0)
    do = st.number_input("Dissolved Oxygen (mg/L)", value=6.5)

with col2:
    cdc = st.number_input("Conductivity (µS/cm)", value=500.0)
    turbidity = st.number_input("Turbidity (NTU)", value=10.0)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Microplastic Concentration"):

    # Create input dataframe
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

    # Scale using TRAINED scaler
    X_scaled = scaler.transform(X_input)

    # Predict
    prediction = model.predict(X_scaled)[0]

    # Output
    st.success("✅ Prediction Successful")
    st.metric(
        label="Predicted Microplastic Concentration",
        value=f"{prediction:.2f} particles/L"
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Developed for CDS590 Practicum | Microplastic Prediction under Data Scarcity"
)
