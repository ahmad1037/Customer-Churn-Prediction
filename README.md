# 📉 Customer Churn Prediction

![CI Status](https://github.com/<ahmad1037>/<Customer-Churn-Prediction>/actions/workflows/ci.yml/badge.svg)

## Customer Churn Prediction

End-to-end Machine Learning project that predicts customer churn using Gradient Boosting and provides predictions through a FastAPI REST API with a Streamlit dashboard.

## Features

- End-to-end ML pipeline
- Business-focused EDA
- Feature Engineering
- Hyperparameter Tuning
- SHAP Explainability
- FastAPI Backend
- Streamlit Dashboard
- Docker Support
- Unit Testing
- CI/CD

## Tech Stack

| Category         | Technologies     |
| ---------------- | ---------------- |
| Language         | Python           |
| ML               | Scikit-Learn     |
| Data             | Pandas, NumPy    |
| Visualization    | Matplotlib, SHAP |
| API              | FastAPI          |
| Frontend         | Streamlit        |
| Testing          | Pytest           |
| Containerization | Docker           |
| CI               | GitHub Actions   |

## Dataset

IBM Telco Customer Churn Dataset (Enhanced)

7043 Customers

33 Features

Binary Classification

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|:--------:|:---------:|:------:|:--------:|:-------:|
| **Gradient Boosting** | **0.8042** | **0.6832** | 0.4911 | 0.5714 | **0.8548** |
| Logistic Regression | 0.8013 | 0.6564 | **0.5302** | **0.5866** | 0.8420 |
| XGBoost | 0.7682 | 0.5783 | 0.4733 | 0.5205 | 0.8357 |
| Random Forest | **0.8042** | **0.7056** | 0.4520 | 0.5510 | 0.8335 |
| Decision Tree | 0.7729 | 0.5837 | 0.5089 | 0.5437 | 0.6887 |

> **Selected Model:** **Gradient Boosting**
>
> Gradient Boosting was chosen for deployment because it achieved the **highest ROC-AUC (0.8548)** while maintaining strong overall accuracy and balanced classification performance.

## Architecture

![Architecture Diagram](images/User%20Streamlit%20FastAPI-2026-07-30-053445.png)

## Screenshots

![alt text](reports/app_demo.gif)

## Installation

</> Bash

    git clone https://github.com/ahmad1037/customer-churn-prediction.git

    cd customer-churn-prediction

    python -m venv .venv

    source .venv/bin/activate

    # Windows
    # .venv\Scripts\activate

    pip install -r requirements.txt


## Run API

uvicorn app.main:app --reload


## Run Streamlit

streamlit run frontend/app.py


## Docker

docker compose up --build


## Author

Ahmad Abbas

GitHub: https://github.com/ahmad1037

LinkedIn: www.linkedin.com/in/ahmad1037

