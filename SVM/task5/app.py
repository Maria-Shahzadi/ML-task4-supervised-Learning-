import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# PAGE
st.set_page_config(page_title="Cancer Predictor", layout="wide")

# LOAD
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

features = joblib.load("features.pkl")

# STYLE
st.markdown("""
<style>
h1 {color:#38bdf8; text-align:center;}
.stButton>button {
    background:#22c55e; color:white;
    border-radius:10px; height:3em; width:100%;
}
</style>
""", unsafe_allow_html=True)

st.title("🧬 Breast Cancer Prediction System")

st.write("Enter patient details below:")

# SIDEBAR INPUTS
inputs = []

for f in features:
    val = st.sidebar.slider(f, 0.0, 100.0, 10.0)
    inputs.append(val)

# CONVERT
input_array = np.array([inputs])

# SCALE
input_scaled = scaler.transform(input_array)

# PREDICT
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("🚨 Malignant Tumor")
        else:
            st.success("✅ Benign Tumor")

    with col2:
        st.metric("Cancer Probability", f"{prob:.2f}")

# SIMPLE VISUAL
st.subheader("📊 Input Overview")

fig, ax = plt.subplots()
ax.bar(features, inputs)
plt.xticks(rotation=45)

st.pyplot(fig)