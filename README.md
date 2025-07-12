# ai_tools_assignment-WK5





# AI Tools Assignment - Streamlit Web App

## Overview
This is a simple Streamlit app that uses a trained TensorFlow model to classify handwritten digits from the MNIST dataset.

## Features
- Upload a PNG/JPG image of a handwritten digit
- The app processes the image, makes a prediction, and displays the result
- Uses a Convolutional Neural Network trained on MNIST

## Files
- `app_streamlit.py`: The main Streamlit app
- `mnist_model.h5`: The pre-trained model (save this from your notebook)

## Setup Instructions

1. Install required packages:
    ```bash
    pip install streamlit tensorflow pillow numpy
    ```

2. Save the trained model:
    ```python
    model.save("mnist_model.h5")
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app_streamlit.py
    ```

## Notes
- Ensure uploaded images are 28x28 grayscale or they'll be resized.
- For best results, use white background with black digits.
- Can be deployed to [Streamlit Cloud](https://streamlit.io/cloud)

## Author
Bravin Ouma
