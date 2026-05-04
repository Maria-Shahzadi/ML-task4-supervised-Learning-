# ================================
# IMPORTS
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide"
)

# ================================
# CUSTOM CSS (BEAUTIFUL UI)
# ================================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton>button {
    background-color: #22c55e;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.sidebar .sidebar-content {
    background-color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD MODELS
# ================================
rf_model = joblib.load("random_forest.pkl")
dt_model = joblib.load("decision_tree.pkl")

# ================================
# LOAD DATASET
# ================================
data = pd.read_csv("titanic.csv")

# ================================
# TITLE
# ================================
st.title("🚢 Titanic Survival Prediction Dashboard")

st.write("Compare Decision Tree vs Random Forest and predict survival")

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("Passenger Details")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Decision Tree"]
)

Pclass = st.sidebar.selectbox("Passenger Class", [1,2,3])
Gender = st.sidebar.selectbox("Gender", ["male","female"])
Age = st.sidebar.slider("Age", 1, 80, 25)
SibSp = st.sidebar.slider("Siblings/Spouses", 0, 8, 0)
Parch = st.sidebar.slider("Parents/Children", 0, 6, 0)
Fare = st.sidebar.slider("Fare", 0.0, 500.0, 50.0)
Embarked = st.sidebar.selectbox("Embarked", ["C","Q","S"])

# ================================
# ENCODING (same as training)
# ================================
Gender = 1 if Gender == "male" else 0
embark_map = {"C":0, "Q":1, "S":2}
Embarked = embark_map[Embarked]

input_data = np.array([[Pclass, Gender, Age, SibSp, Parch, Fare, Embarked]])

# ================================
# SELECT MODEL
# ================================
model = rf_model if model_choice == "Random Forest" else dt_model

# ================================
# PREDICTION
# ================================
st.subheader("🔮 Prediction")

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.success(f"🎉 Survived (Probability: {prob:.2f})")
        else:
            st.error(f"💀 Did Not Survive (Probability: {prob:.2f})")

    with col2:
        st.metric("Survival Probability", f"{prob:.2f}")

# ================================
# FEATURE IMPORTANCE
# ================================
st.subheader("📊 Feature Importance")

features = ["Pclass","Gender","Age","SibSp","Parch","Fare","Embarked"]

importances = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_title(f"{model_choice} Feature Importance")

st.pyplot(fig)

# ================================
# DATASET PREVIEW
# ================================
st.subheader("📂 Dataset Preview")
st.dataframe(data.head())

# ================================
# BASIC INSIGHTS
# ================================
st.subheader("📈 Survival Insights")

# Survival by gender
gender_survival = data.groupby("Gender")["Survived"].mean()
st.bar_chart(gender_survival)

# Survival by class
class_survival = data.groupby("Pclass")["Survived"].mean()
st.bar_chart(class_survival)

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("✨ Built with Streamlit | ML Project Dashboard")