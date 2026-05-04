# ================================
# 1. IMPORT LIBRARIES
# ================================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ================================
# 2. PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Maria AI Health Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ================================
# 3. LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ================================
# 4. HEADER
# ================================
st.markdown("""
# 🧠 Maria's AI Breast Cancer Prediction System
### Smart Diagnosis using Machine Learning
""")

st.markdown("---")

# ================================
# 5. FEATURE NAMES
# ================================
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# ================================
# 6. SIDEBAR INPUT
# ================================
st.sidebar.header("🔧 Patient Data Input")

inputs = []
for feature in feature_names:
    val = st.sidebar.slider(feature, 0.0, 50.0, 10.0)
    inputs.append(val)

input_data = np.array(inputs).reshape(1, -1)

# ================================
# 7. MAIN LAYOUT
# ================================
col1, col2 = st.columns([1,1])

# ================================
# LEFT PANEL
# ================================
with col1:
    st.subheader("📊 Patient Data Overview")
    df = pd.DataFrame(input_data, columns=feature_names)
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Feature Importance (Model Weights)")
    
    weights = model.coef_[0]
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Weight": weights
    }).sort_values(by="Weight", key=abs, ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Weight"])
    ax.invert_yaxis()
    ax.set_title("Top 10 Influential Features")
    st.pyplot(fig)

# ================================
# RIGHT PANEL
# ================================
with col2:
    st.subheader("🔍 Prediction Result")

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    # Result Display
    if prediction == 1:
        st.success("✅ Benign (Safe)")
    else:
        st.error("⚠️ Malignant (Cancer Detected)")

    st.metric("Benign Probability", f"{prob[1]*100:.2f}%")
    st.metric("Malignant Probability", f"{prob[0]*100:.2f}%")

    # ================================
    # PROBABILITY GRAPH
    # ================================
    st.subheader("📊 Probability Distribution")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Malignant", "Benign"], prob)
    ax2.set_ylabel("Probability")
    ax2.set_title("Prediction Confidence")
    st.pyplot(fig2)

# ================================
# EXTRA FEATURES
# ================================
st.markdown("---")

st.subheader("💡 AI Insights")

if prediction == 1:
    st.info("The tumor shows characteristics of a non-cancerous pattern.")
else:
    st.warning("The tumor shows patterns similar to malignant cases. Further medical testing is recommended.")

# ================================
# FOOTER
# ================================
st.markdown("""
---
### 👩‍💻 Developed by Maria  
AI & Data Science Dashboard  
""")