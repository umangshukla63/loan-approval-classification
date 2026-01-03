# Loan Default Risk Prediction — Machine Learning Project

This project builds an end-to-end machine learning pipeline to predict whether a borrower is likely to default on a loan.  
The goal is to help financial institutions make smarter, risk-aware loan approval decisions.

The project follows a complete data science workflow — from business framing to model explainability.



## 🔍 Problem Statement

Approving loans to customers who later default creates financial losses.  
Rejecting good customers leads to lost business.

The objective is to build a model that:

- predicts the probability of default
- minimizes costly false negatives (approving risky borrowers)
- remains interpretable and aligned with business logic



## 📊 Dataset

- ~50,000 rows  
- ~20 features  
- demographic, income, credit history, debt ratios and loan attributes  

##  Project Steps

### 1. Business Understanding
Defined the problem from a banking perspective, including risk trade-offs and evaluation priorities.

### 2. Data Understanding & EDA
- missing values, duplicates, datatypes
- class distribution
- visual analysis of income, credit score, debt ratios, and defaults
- correlations and key insights

### 3. Modeling Pipeline
Built a clean scikit-learn pipeline using:

- `ColumnTransformer`
- `StandardScaler`
- `OneHotEncoder`
- Logistic Regression, Decision Tree, Random Forest

### 4. Feature Engineering
Added domain-driven financial features including:

- credit utilization  
- income per year employed  
- credit score buckets  
- high-risk financial flags  

These significantly improved model performance.

### 5. Model Evaluation
Evaluated using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Special priority was given to **reducing false negatives**.

### 6. Hyperparameter Tuning + Threshold Optimization
- GridSearchCV on Random Forest
- Adjusted probability threshold to better control risk

### 7. Explainability
- Feature importance
- Business interpretation of results



## Final Model

**Random Forest (tuned) + optimized decision threshold**

Key characteristics:

- strong ROC-AUC  
- high recall for default class  
- balanced precision  
- aligned with business priorities  



## Key Insights

- Debt-to-income ratio and credit score are the strongest predictors
- Risk increases with high utilization and short credit history
- Loan affordability matters more than absolute loan amount
- Threshold tuning is critical in credit-risk applications





