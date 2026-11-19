# Telco Customer Churn Prediction

IT3091 Machine Learning group project (4 members) at SLIIT. The project predicts which telecom customers are likely to leave (churn), using Logistic Regression, SVM, Random Forest and XGBoost.

## Dataset
Telco Customer Churn dataset (7,043 customers), the IBM sample dataset available on Kaggle.

## Tools
Python, pandas, NumPy, scikit-learn, XGBoost, SMOTE

## Results
The tuned XGBoost model (SMOTE + GridSearchCV) reached 78.6% recall for churners and a ROC-AUC of 0.856.

## My Contribution (Member 2)
Data preprocessing and feature engineering:
- Created new features: Tenure_in_Years and Total_Services_Subscribed
- One-hot encoded categorical variables (31 final features)
- 80/20 train/test split
- Standardised numeric features with StandardScaler, fitted on training data only to prevent data leakage

## Files
- `telco_churn_prediction.ipynb`: full project notebook
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: dataset
