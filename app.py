import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Glasses Classifier",
    page_icon="🕶️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS ----
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: #111;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🕶️ Glasses Classifier")
st.write("Upload a face image to check if the person is wearing glasses.")

# ---- LOAD MODEL ----
@st.cache_resource
def load_glasses_model():
    return load_model("glasses_classifier.keras")

model = load_glasses_model()

# ---- IMAGE UPLOAD ----
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = img.resize((64, 64))  # match your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    pred = model.predict(img_array)
    # Assuming model outputs 0 = glasses, 1 = no glasses
    if pred[0][0] >= 0.5:
        st.success("❌ Person is NOT wearing glasses.")
    else:
        st.info("🕶️ Person is wearing glasses!")
