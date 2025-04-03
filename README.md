# Heart Disease Prediction App

This repository contains a heart disease prediction application built with TensorFlow and Streamlit. The model uses a DenseNet neural network architecture to predict the risk of heart disease based on various health parameters.

## Files in this Repository

- `app.py` - Streamlit application for deploying on Hugging Face Spaces
- `colab_app.py` - Google Colab notebook version with additional analysis features
- `requirements.txt` - Required dependencies for deployment
- `heart_disease_new/heart_disease.csv` - Dataset used for training

## How to Use

### Running the Streamlit App Locally

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Access the application in your web browser at `http://localhost:8501`

### Running in Google Colab

1. Upload `colab_app.py` to Google Colab
2. Upload the dataset `heart_disease.csv` when prompted
3. Run all cells to perform the full analysis and create the model

### Deploying on Hugging Face Spaces

1. Create a new Space on Hugging Face Spaces
2. Select Streamlit as the SDK
3. Upload all files from this repository
4. The app will be automatically deployed

## Model Details

The heart disease prediction model uses a DenseNet neural network architecture, which is particularly effective for this type of classification task. Key features include:

- Dense connectivity pattern for better feature propagation and reuse
- Batch normalization and dropout for regularization
- L2 regularization to prevent overfitting
- Class weighting for handling imbalanced data
- Early stopping to prevent overfitting

## Dataset

The dataset contains various health parameters related to heart disease, including:

- Age, Sex, and other demographic factors
- Various clinical measurements and test results
- Medical history and lifestyle factors

## Acknowledgments

This project uses the TensorFlow framework and Streamlit for the web interface. The DenseNet architecture is inspired by the paper ["Densely Connected Convolutional Networks"](https://arxiv.org/abs/1608.06993) by Huang et al. 