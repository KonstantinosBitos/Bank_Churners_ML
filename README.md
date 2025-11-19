# Bank Customer Churn Prediction

## Project Overview
This project predicts whether a bank customer is likely to **exit (churn)** using demographic and financial data. It demonstrates **data preprocessing, feature engineering, machine learning modeling, evaluation, and threshold optimization**. 

The main goal is to identify potential churners so that the bank can take proactive retention measures.

---

## Dataset
The dataset contains customer-level information, including:

- `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- Categorical features: `Gender`, `Geography`
- Target variable: `Exited` (0 = stayed, 1 = churned)

Dataset source: https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers/data
---

## Feature Engineering
Key transformations performed:

- **Age Groups:** `Age` divided into 6 quantile-based groups  
- **Credit Score Groups:** `CreditScore` divided into 8 quantile-based groups  
- **Balance Score Groups:** `Balance` ranked and grouped into 5 bins (handles many zero balances)  
- **Estimated Salary Score Groups:** `EstimatedSalary` divided into 10 bins  
- **BalanceSalaryRatio:** `Balance / (EstimatedSalary + 1)`  
- **Customer Lifetime Percentage:** `(Tenure / Age) * 100`  
- **HighSalary:** Binary feature indicating if `EstimatedSalary` is above median  
- One-hot encoding applied to `Geography`, `Gender` mapped to 0/1

---

## Models
The project includes three models:

1. **Random Forest Classifier**  
2. **Gradient Boosting Classifier**  
3. **Tuned Gradient Boosting Classifier** (optimized hyperparameters and threshold)

**Class imbalance** was addressed using `class_weight` for Random Forest and `scale_pos_weight` for Gradient Boosting. Thresholds were optimized to balance **precision and recall**.

---

## Evaluation Metrics
Models were evaluated using:

- **Classification Report**: Precision, Recall, F1-score  
- **ROC-AUC Score**  
- **ROC Curve Plots**  
- **Feature Importance Visualization**

Example results (Gradient Boosting Tuned):
          precision    recall  f1-score   support
       0       0.92      0.86      0.89      1593
       1       0.56      0.70      0.62       407
accuracy                           0.83      2000
macro avg 0.74 0.78 0.76 2000
weighted avg 0.85 0.83 0.84 2000

**ROC-AUC:** 0.87

---

## Visualizations
- ROC Curves (Random Forest vs Gradient Boosting)  
- Feature Importance Plots  
- Threshold optimization analysis

> All plots are included in the notebook.



