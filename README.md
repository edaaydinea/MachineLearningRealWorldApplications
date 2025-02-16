# Machine Learning: 6 Real-World Applications

This repository covers practical machine learning scenarios, each illustrating a unique real-world use case.

## Table of Contents

- [Machine Learning: 6 Real-World Applications](#machine-learning-6-real-world-applications)
  - [Table of Contents](#table-of-contents)
  - [1. Breast Cancer Classification](#1-breast-cancer-classification)
  - [2. Fashion Class Classification](#2-fashion-class-classification)
  - [3. Directing Customers to Subscription Through App Behavior Analysis](#3-directing-customers-to-subscription-through-app-behavior-analysis)
  - [4. Minimizing Churn Rate Through Analysis of Financial Habits](#4-minimizing-churn-rate-through-analysis-of-financial-habits)
  - [5. Predicting the Likelihood of E-Signing a Loan Based on Financial History](#5-predicting-the-likelihood-of-e-signing-a-loan-based-on-financial-history)
  - [6. Credit Card Fraud Detection](#6-credit-card-fraud-detection)

## 1. Breast Cancer Classification

A dataset from the `sklearn` library is used to predict whether breast cancer is benign or malignant. The workflow involves data loading, preprocessing, and splitting into training and testing sets. Model performance is evaluated using accuracy, confusion matrices, and classification reports.

Notebook:

- [BreastCancerClassification.ipynb](./BreastCancerClassification/BreastCancerClassification.ipynb)

## 2. Fashion Class Classification

The Fashion MNIST dataset is used to classify images of clothing items into 10 classes. The workflow involves data loading, preprocessing, and splitting into training and testing sets. Model performance is evaluated using accuracy, confusion matrices, and classification reports.

Notebook:

- [FashionClassClassification.ipynb](./FashionClassClassification/FashionClassClassification.ipynb)

## 3. Directing Customers to Subscription Through App Behavior Analysis

This project analyzes user behavior within an app to predict the likelihood of converting free users to paid subscribers. The workflow involves data collection, preprocessing, feature engineering, and model training. Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

Notebook:

- [DataPreprocessing.ipynb](./FinTechCaseStudies/notebooks/01_data_preprocessing.ipynb)
- [Modeling.ipynb](./FinTechCaseStudies/notebooks/02_model_training.ipynb)

## 4. Minimizing Churn Rate Through Analysis of Financial Habits

This project aims to reduce customer churn by analyzing financial habits and predicting the likelihood of customers leaving a subscription service. The workflow involves data loading, preprocessing, feature engineering, and model training. Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

Notebook:

- [DataPreprocessing.ipynb](./ChurnAnalysis/notebooks/01_data_preprocessing.ipynb)
- [Modeling.ipynb](./ChurnAnalysis/notebooks/02_modeling.ipynb)

## 5. Predicting the Likelihood of E-Signing a Loan Based on Financial History

This project predicts the likelihood of a customer e-signing a loan based on their financial history. The workflow involves data loading, preprocessing, feature engineering, and model training. Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

Notebook:

- [Pipeline.ipynb](./FinancialAnalysis/notebooks/pipeline.ipynb)

## 6. Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using various machine learning models. The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. The workflow involves data loading, preprocessing, feature engineering, and model training. Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The model is performed using a Random Forest Classifier, Decision Tree, XGBoost, LightGBM, and deep learning models. As a result, the deep learning model with SMOTE sampling technique has the best performance with an F1-score of 1.0. 

Notebook:

- [Pipeline.ipynb](./CreditCardFraudDetection/notebooks/pipeline.ipynb)