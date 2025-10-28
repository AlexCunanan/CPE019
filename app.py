import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("rps_cnn.keras")
class_names = ['rock', 'paper', 'scissors']

st.title("✂️ Rock-Paper-Scissors Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((100,100))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.3f}")
    st.image(img, caption="Uploaded Image", use_column_width=True)
