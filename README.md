# 🏢 Prediction of Company Bankruptcy using Machine Learning

This project uses supervised learning algorithms to predict whether a company will go bankrupt based on financial indicators.

---

## 📁 Directory Structure

Predict_Of_Company_Bankruptcy_Using_ML/
│
├── Data/
│ └── company_top4_features.csv
│ └── train_data.csv
│ └── test_data.csv
│
├── Training/
│ └── preprocess_data.ipynb
│ └── training_notebook.ipynb
│
├── Evaluation/
│ └── evaluation_and_tuning.ipynb
│ └── best_model_saving.ipynb
│
├── App/
│ ├── app.py
│ ├── model/
│ │ └── model.pkl
│ ├── templates/
│ │ └── index.html
│ ├── static/
│ └── css/
│ └── js/
│
├── requirements.txt
├── python_version.txt
└── README.md

---

## 💡 Project Overview

- **Goal**: Predict if a company will fail or stay alive using financial data.
- **Dataset**: Cleaned and balanced version of an original bankruptcy dataset.
- **Features Used**: Top 4 features identified using feature importance:
  - `X1`: Current Ratio
  - `X4`: Total Asset Turnover
  - `X6`: Retained Earnings to Total Assets
  - `X10`: Net Income to Total Assets

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest ✅ (Best: ~64.5% accuracy)
- AdaBoost
- Gradient Boosting

> **Best Model**: Random Forest

---

## 🚀 How to Run

1. **Install dependencies**  
```bash
pip install -r requirements.txt
