import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ================================
# 1. PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Maria AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ================================
# 2. THEME SWITCH (LIGHT / DARK)
# ================================
theme = st.sidebar.radio("🎨 Choose Theme", ["Light Mode", "Dark Mode"])

if theme == "Dark Mode":
    st.markdown("""
        <style>
        body { background-color: #0f172a; color: white; }
        </style>
    """, unsafe_allow_html=True)

# ================================
# 3. LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ================================
# 4. SIDEBAR NAVIGATION
# ================================
menu = st.sidebar.selectbox(
    "📌 Navigation",
    ["🏠 Home", "📊 Predict Customer Attrition", "📈 Model Insights"]
)

# ================================
# HOME PAGE
# ================================
if menu == "🏠 Home":
    st.title("🧠 Maria AI Customer Analytics Dashboard")
    st.write("Welcome to a smart ML-powered system for predicting customer attrition.")
    st.info("Use sidebar to navigate")

# ================================
# PREDICTION PAGE
# ================================
elif menu == "📊 Predict Customer Attrition":
    st.title("📊 Customer Attrition Prediction")

    st.subheader("Enter Customer Data")

    # Example inputs (you can expand later)
    age = st.slider("Customer Age", 18, 100, 40)
    credit_limit = st.number_input("Credit Limit", 0.0, 50000.0, 5000.0)
    total_trans = st.number_input("Total Transactions", 0, 200, 50)
    inactive = st.slider("Months Inactive", 0, 12, 2)

    if st.button("🚀 Predict Now"):
        
        # simple feature array (adjust based on training columns)
        input_data = np.array([[age, credit_limit, total_trans, inactive]])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        st.subheader("🔍 Result")

        if prediction == 1:
            st.error("⚠️ Customer Will Leave (Attrition)")
        else:
            st.success("✅ Customer Will Stay")

        st.metric("Stay Probability", f"{prob[0]*100:.2f}%")
        st.metric("Leave Probability", f"{prob[1]*100:.2f}%")

        # ================================
        # PROBABILITY GRAPH
        # ================================
        st.subheader("📊 Prediction Probability")

        fig, ax = plt.subplots()
        ax.bar(["Stay", "Leave"], prob)
        st.pyplot(fig)

# ================================
# MODEL INSIGHTS PAGE
# ================================
elif menu == "📈 Model Insights":
    st.title("📈 Model Performance Dashboard")

    st.write("Below are model evaluation insights (example simulated section).")

    # Fake sample curves (replace with real test data if needed)
    y_test = np.array([0,1,0,1,0,1,1,0])
    y_prob = np.array([0.2,0.8,0.3,0.7,0.1,0.9,0.6,0.4])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader("📈 ROC Curve")

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    st.pyplot(fig)

    # Precision Recall
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    st.subheader("📉 Precision-Recall Curve")

    fig2, ax2 = plt.subplots()
    ax2.plot(recall, precision)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    st.pyplot(fig2)

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("👩‍💻 Developed by **Maria** | AI & Data Science Dashboard")