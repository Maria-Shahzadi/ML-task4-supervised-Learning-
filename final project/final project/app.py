import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# ================================
# LOAD MODEL
# ================================
model = pickle.load(open("best_model.pkl", "rb"))

# ================================
# SIDEBAR NAVIGATION
# ================================
menu = st.sidebar.radio(
    "📌 Menu",
    ["🏠 Home", "🔮 Predict", "📊 Charts"]
)

# ================================
# HOME PAGE
# ================================
if menu == "🏠 Home":
    st.title("🚢 Titanic Survival Dashboard")
    st.write("Welcome to your ML prediction system 💙")

    st.success("Model is loaded and ready to predict!")

# ================================
# PREDICTION PAGE
# ================================
elif menu == "🔮 Predict":

    st.title("🔮 Survival Prediction")

    Pclass = st.selectbox("Passenger Class", [1,2,3])
    Gender = st.selectbox("Gender", ["male","female"])   # FIXED NAME
    Age = st.slider("Age", 1, 80, 25)
    SibSp = st.slider("Siblings/Spouses", 0, 8, 0)
    Parch = st.slider("Parents/Children", 0, 6, 0)
    Fare = st.slider("Fare", 0.0, 500.0, 50.0)
    Embarked = st.selectbox("Embarked", ["C","Q","S"])

    # ENCODING
    Gender = 1 if Gender == "male" else 0
    Embarked = {"C":0,"Q":1,"S":2}[Embarked]

    input_data = np.array([[Pclass, Gender, Age, SibSp, Parch, Fare, Embarked]])

    if st.button("🚀 Predict Now"):

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        col1, col2 = st.columns(2)

        with col1:
            if pred == 1:
                st.balloons()   # 🎉 CELEBRATION
                st.success("🎉 SURVIVED")
            else:
                st.error("💀 DID NOT SURVIVE")

        with col2:
            st.metric("Survival Probability", f"{prob:.2f}")

# ================================
# CHARTS PAGE
# ================================
elif menu == "📊 Charts":

    st.title("📊 Data Visualization")

    data = pd.DataFrame({
        "Feature": ["Pclass","Age","SibSp","Parch","Fare"],
        "Value": [3,25,1,0,50]
    })

    fig, ax = plt.subplots()
    ax.bar(data["Feature"], data["Value"])
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.info("Simple demo chart (you can expand later with real dataset graphs)")