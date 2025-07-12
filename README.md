# ai_tools-WK5

# AI Tools Assignment - Mastering the AI Toolkit  

#Project Overview

This project demonstrates mastery of popular AI tools through theoretical understanding, hands-on implementation, and ethical reflection. The core focus is to bridge classical ML, deep learning, NLP, and deployment using Streamlit.

#Sections Covered

#1. Theoretical Understanding
- Differences between **TensorFlow** and **PyTorch**
- Use cases of **Jupyter Notebooks** in data science workflows
- How **spaCy** enhances natural language processing tasks
- Comparative analysis: **Scikit-learn** vs **TensorFlow**

#2. Practical Implementation

#Task 1: Classical Machine Learning with Scikit-learn
- Dataset: Iris
- Model: Decision Tree Classifier
- Metrics: Accuracy, Precision, Recall

#Task 2: Deep Learning with TensorFlow
- Dataset: MNIST (handwritten digits)
- Model: Convolutional Neural Network (CNN)
- Visualization of predictions included
- Model saved as `mnist_model.h5` for deployment

#Task 3: Natural Language Processing with spaCy & TextBlob
- Named Entity Recognition (NER)
- Sentiment Analysis

## Bonus: Streamlit Deployment

A web app built with **Streamlit** to upload digit images and classify them using the trained CNN model.

#File: `app_streamlit.py`
- Uploaded image of a digit
- Real-time prediction from the TensorFlow model
- Image preprocessing and digit classification handled inside the app

#Ethical Considerations

- Recognized potential biases in training datasets
- Reflected on fairness, cultural representation, and safe deployment practices
- Proposed solutions like rule-based filtering and fairness indicators


#How to Run the Project

1. Clone the repo and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run app_streamlit.py
    ```

3. (Optional) Re-train and save the CNN model:
    ```python
    model.save("mnist_model.h5")
    ```

---

#Deliverables

- ✅ Python code for ML, DL, and NLP
- ✅ Streamlit Web App (`app_streamlit.py`)
- ✅ Ethics reflections and troubleshooting
- ✅ TXT Report
- ✅ 3-Minute Group Video Presentation


# AI Tools Assignment - Streamlit Web App

#Overview
This is a simple Streamlit app that uses a trained TensorFlow model to classify handwritten digits from the MNIST dataset.

## Features
- Upload a PNG/JPG image of a handwritten digit
- The app processes the image, makes a prediction, and displays the result
- Uses a Convolutional Neural Network trained on MNIST

#Files
- `app_streamlit.py`: The main Streamlit app
- `mnist_model.h5`: The pre-trained model (save this from your notebook)

#Setup Instructions

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

RUN   pip install -r requirements.txt
TO ISNTALL REQUIREMENTS

## Author

Bravin Ouma

Full Stack Engineer | AI Tools Enthusiast | PLP Cohort  
