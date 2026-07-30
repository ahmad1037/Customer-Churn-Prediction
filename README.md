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

## Model Performance

                 Model  Accuracy  Precision    Recall        F1   ROC-AUC
0  Logistic Regression  0.801325   0.656388  0.530249  0.586614  0.842022
1        Decision Tree  0.772942   0.583673  0.508897  0.543726  0.688727
2        Random Forest  0.804163   0.705556  0.451957  0.550976  0.833527
3    Gradient Boosting  0.804163   0.683168  0.491103  0.571429  0.854803
4              XGBoost  0.768212   0.578261  0.473310  0.520548  0.835712

## Architecture
                Streamlit

                    │

                    ▼

              FastAPI API

                    │

                    ▼

        Preprocessing Pipeline

                    │

                    ▼

          Trained ML Model

                    │

                    ▼

          Predicted House Price

![Architecture Diagtram](images/User%20Streamlit%20FastAPI-2026-07-30-053445.png)

## Screenshots

![App Screenshot](reports/Customer%20churn%20Prediction.gif)


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

