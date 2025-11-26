import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# ============================
# Load Model
# ============================
MODEL_PATH = "glasses_classifier.keras"
model = load_model(MODEL_PATH)

# ============================
# Streamlit Page Config
# ============================
st.set_page_config(
    page_title="Glasses vs No-Glasses Classifier",
    page_icon="🕶️",
    layout="wide"
)

# ============================
# Custom UI Styling
# ============================
st.markdown("""
<style>
.stApp {
    background-color: #1a1a1a;
    color: white;
    font-family: 'Poppins', sans-serif;
}
.title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    color: #00e5ff;
    text-shadow: 0px 0px 15px #00e5ff;
}
.pred {
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    margin-top: 15px;
}
.conf {
    font-size: 22px;
    text-align: center;
    color: #ffd700;
}
</style>
""", unsafe_allow_html=True)

# ============================
# Title
# ============================
st.markdown("<h1 class='title'>🕶️ Glasses vs No-Glasses Classifier</h1>", unsafe_allow_html=True)
st.write("Upload one or more images to check whether the person is wearing **glasses** or **no glasses**.")

# ============================
# File Uploader
# ============================
uploaded_files = st.file_uploader(
    "Upload face images (JPG / PNG)",
    type=["jpg", "png"],
    accept_multiple_files=True
)

# ============================
# Prediction Function
# ============================
def predict_image(img):
    img = img.resize((64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        label = "🕶️ Glasses"
        confidence = pred * 100
        color = "#00e5ff"
    else:
        label = "👀 No Glasses"
        confidence = (1 - pred) * 100
        color = "#ff5e5e"

    return label, confidence, color

# ============================
# Display Predictions
# ============================
if uploaded_files:
    for uploaded_file in uploaded_files:
        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption=uploaded_file.name, use_column_width=True)

        with col2:
            label, conf, color = predict_image(img)

            st.markdown(
                f"<p class='pred' style='color:{color}'>{label}</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p class='conf'>Confidence: {conf:.2f}%</p>",
                unsafe_allow_html=True
            )
            st.progress(int(conf))

# ============================
# Footer
# ============================
st.markdown("""
<hr>
<p style='text-align:center; color:#888;'>© 2025 Glasses Classifier | Built with ❤️ using Streamlit & TensorFlow</p>
""", unsafe_allow_html=True)
