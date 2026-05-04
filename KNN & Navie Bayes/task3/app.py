import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# ================================
# 1. PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Maria AI Dashboard",
    layout="wide",
    page_icon="🧠"
)

# ================================
# 2. DARK / LIGHT MODE FIX
# ================================
theme = st.sidebar.radio("🎨 Theme Mode", ["Light Mode", "Dark Mode"])

if theme == "Dark Mode":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
        color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================
# 3. LOAD MODEL
# ================================
knn_model = pickle.load(open("knn_model.pkl", "rb"))

# ================================
# 4. LOAD DATA
# ================================
digits = load_digits()

st.title("🧠 Maria AI - Digit Classification Dashboard")

st.markdown("Predict handwritten digits using KNN model")

# ================================
# 5. SIDEBAR INPUT (SIMPLIFIED)
# ================================
st.sidebar.header("✍️ Input Controls")

sample_index = st.sidebar.slider(
    "Choose Sample Digit",
    0,
    len(digits.data)-1,
    0
)

# get real digit sample (NO 64 sliders needed)
input_data = digits.data[sample_index].reshape(1, -1)

# ================================
# 6. PREDICTION
# ================================
pred = knn_model.predict(input_data)[0]

st.subheader("🔍 Prediction Result")

if st.button("🚀 Predict Digit"):
    st.success(f"Predicted Digit: {pred}")

# ================================
# 7. IMAGE VISUALIZATION (GRAPH FIX)
# ================================
st.subheader("📊 Digit Visualization")

fig, ax = plt.subplots()
ax.imshow(digits.images[sample_index], cmap="gray")
ax.set_title(f"Actual Digit: {digits.target[sample_index]}")
ax.axis("off")

st.pyplot(fig)

# ================================
# 8. PCA VISUALIZATION GRAPH
# ================================
st.subheader("📈 Dataset Distribution (PCA)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(digits.data)

fig2, ax2 = plt.subplots()
scatter = ax2.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=digits.target,
    cmap="tab10",
    alpha=0.6
)

ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.set_title("Digits Distribution")

st.pyplot(fig2)

# ================================
# 9. FOOTER
# ================================
st.markdown("---")
st.markdown("👩‍💻 Developed by Maria | AI Dashboard Project")