# Customer-Churn-Prediction-System
Predicts telecom customer churn using machine learning (XGBoost, Scikit-learn) with an interactive Streamlit UI

A machine learning web app that predicts whether a telecom customer is likely to churn,
built using Python, Scikit-learn, XGBoost, and Streamlit.

## Project Overview

Customer churn is when a customer stops using a company's service.
This project uses historical telecom customer data to train a classification 
model that predicts churn probability and provides actionable business recommendations.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit

## Dataset

Telco Customer Churn Dataset from Kaggle  
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## How to Run Locally

1. Clone the repository
   git clone https://github.com/yourusername/customer-churn-prediction.git

2. Install dependencies
   pip install -r requirements.txt

3. Run the app
   streamlit run app.py

## Live Demo

https://your-app-link.streamlit.app

## Project Structure

customer-churn-prediction/
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md

## Model Performance

| Model              | Accuracy |
|--------------------|----------|
| Logistic Regression| ~79%     |
| Random Forest      | ~82%     |
| XGBoost            | ~84%     |

## Author

Your Name  
Edunet Foundation IBM AI/ML Internship — 2025
