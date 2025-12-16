# Bank Customer Churn Prediction

## Overview
This project predicts whether a bank customer is likely to **exit (churn)** using demographic and financial data. It demonstrates **data preprocessing, feature engineering, machine learning modeling, evaluation, and threshold optimization**. 

The main goal is to identify potential churners so that the bank can take proactive retention measures.

---

## Dataset
The dataset was obtained from: https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers/data

| Feature         | Description                                         | Relevance to Churn                                 |
| --------------- | --------------------------------------------------- | -------------------------------------------------- |
| RowNumber       | Corresponds to the record (row) number              | No effect on churn                                 |
| CustomerId      | Contains random values identifying each customer    | No effect on churn                                 |
| Surname         | Customer surname                                    | No effect on churn                                 |
| CreditScore     | Customer credit score                               | Higher credit score → less likely to leave         |
| Geography       | Customer location                                   | Can affect likelihood of leaving                   |
| Gender          | Customer gender                                     | Worth exploring if gender influences churn         |
| Age             | Customer age                                        | Older customers are less likely to leave           |
| Tenure          | Number of years the customer has been with the bank | Longer tenure → more loyalty, less likely to leave |
| Balance         | Account balance                                     | Higher balance → less likely to leave              |
| NumOfProducts   | Number of products the customer has with the bank   | Can influence churn behavior                       |
| HasCrCard       | Whether the customer has a credit card              | Having a credit card → less likely to leave        |
| IsActiveMember  | Whether the customer is an active member            | Active members → less likely to leave              |
| EstimatedSalary | Customer estimated salary                           | Lower salary → more likely to leave                |
| Exited          | Whether the customer left the bank (target)         | Target variable                                    |

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

---

## Visualizations
- ROC Curves (Random Forest vs Gradient Boosting)  
- Feature Importance Plots  

All plots are included in the notebook.

---

## Connect

**Author:** Konstantinos Bitos  
Email: [bitoskostas1@gmail.com](mailto:bitoskostas1@gmail.com)  
Medium: [@bitoskostas1](https://medium.com/@bitoskostas1)

