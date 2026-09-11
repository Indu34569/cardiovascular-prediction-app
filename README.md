# Cardiovascular Disease Prediction

A machine learning web application that predicts whether cardiovascular disease is detected based on user-provided health parameters.

## Project Overview

Cardiovascular diseases are among the major health concerns worldwide. Machine learning can be used to analyze health-related data and assist in predicting the possibility of cardiovascular disease.

This project uses a **Support Vector Machine (SVM)** classification model to make predictions. The trained model and feature scaler are integrated into a **Streamlit** web application.

## Features

- User-friendly Streamlit web interface
- SVM-based classification
- Feature scaling using StandardScaler
- Saved machine learning model using Joblib
- Cardiovascular disease prediction
- Deployed using Streamlit Community Cloud

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Support Vector Machine (SVM)
- StandardScaler
- Joblib
- Streamlit
- GitHub

## Machine Learning Model

The project uses a **Support Vector Machine (SVM)** classifier.

### Workflow

1. Load the dataset
2. Prepare the input features
3. Split the data into training and testing sets
4. Scale the features using StandardScaler
5. Train the SVM classification model
6. Evaluate the model
7. Save the trained model and scaler
8. Use them in the Streamlit application

## Model Performance

The SVM model achieved approximately **73% accuracy** on the test dataset.

## Project Structure

```text
cardiovascular-prediction-app/
│
├── app.py
├── svm_model.pkl
├── scaler.pkl
├── requirements.txt
├── runtime.txt
└── README.md