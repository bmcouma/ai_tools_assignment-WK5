# app_streamlit.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained MNIST model (must be same as trained above and saved)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    return model

model = load_model()

st.title("Digit Recognizer - MNIST")

uploaded_file = st.file_uploader("Upload an image (28x28 grayscale digit)", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    img_array = np.array(image)
    img_array = 255 - img_array  # Invert colors (MNIST style)
    st.image(img_array, caption="Processed Image", width=150)

    input_data = img_array.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    st.write(f"### Predicted Digit: {predicted_class}")
