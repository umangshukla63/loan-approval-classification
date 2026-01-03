# Loan Approval Prediction — Machine Learning Project

This project builds an end-to-end machine learning pipeline to predict whether a loan application should be **approved or rejected** based on customer financial, demographic, and credit information.

The goal is to support loan officers with **data-driven, consistent, and risk-aware** decisions.

---

## Problem Statement

Banks must decide whether a loan application is safe to approve.

Poor decisions cause problems:

- approving financially risky applicants → possible losses  
- rejecting strong applicants → lost revenue + poor customer experience  

The objective is to predict the **probability that a loan will be approved** and use that probability to make better approval decisions.

---

##  Dataset

The dataset contains customer and loan details such as:

- age, occupation, years employed  
- income, savings, existing debt  
- credit score and credit history  
- loan amount, interest rate, loan purpose  
- financial ratios (DTI, LTI, PTI, etc.)

**Target variable**

- 1 → loan approved  
- 0 → loan rejected  

---

##  Project Workflow

### 1. Business Understanding
Defined approval vs rejection trade-offs and realistic banking constraints.

### 2. Data Understanding & EDA
- explored distributions  
- checked class balance  
- analyzed approval rates across income, credit score, debt ratios, etc.

### 3. Modeling Pipeline
Built a clean scikit-learn pipeline using:

- ColumnTransformer  
- StandardScaler  
- OneHotEncoder  

Models trained and compared:

- Logistic Regression  
- Decision Tree  
- Random Forest  

### 4. Feature Engineering
Created domain-driven features:

- credit utilization  
- income per year employed  
- credit score risk buckets  
- loan affordability ratios  

These significantly improved performance.

---

## Evaluation

Metrics used:

- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  

Focus: **avoid approving clearly risky applications while minimizing unnecessary rejections**.

---

## Hyperparameter Tuning & Threshold Optimization
- tuned Random Forest using GridSearchCV  
- adjusted approval threshold to align with risk policy

---

## Explainability
- feature importance visualization  
- interpretation in terms of real lending behavior

---

## Final Model

**Random Forest (tuned) + optimized decision threshold**

✔ strong ROC-AUC  
✔ balanced precision & recall  
✔ risk-aware approval behavior  
✔ interpretable feature influence  

---

## Key Insights

- higher debt-to-income → lower approval chance  
- weak credit score/history → reduced approval likelihood  
- loan affordability matters more than raw loan size  
- engineered features captured financial stress more effectively

---

## About

This project demonstrates an end-to-end workflow for **loan approval modeling**, covering business framing, EDA, modeling, tuning, evaluation, and interpretation.
