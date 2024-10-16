# Regression_Tree_Prediction_Competition
A machine learning project for prediction competitions using Regression Tree algorithms to predict target variables. It emphasizes cross-validation to mitigate overfitting and optimize performance. The current model achieved a Mean Squared Error (MSE) of 0.0090 and an R² score of 0.7234 on the training set.

## Overview
This repository is part of a series of prediction competitions focusing on building and evaluating models using **Regression Tree** algorithms. The goal is to accurately predict target variables based on available features while minimizing overfitting through cross-validation techniques. In this project, the regression tree model achieved the following performance on the training set:
- **MSE**: 0.0090
- **R²**: 0.7234

## Project Structure
The project files are organized as follows:

- **`regression_tree_model.py`**: Python script containing the code for data preprocessing, model training using a regression tree, and implementing cross-validation.
- **`submission.csv`**: CSV file with predictions for the test set, formatted according to competition guidelines.
- **`report.pdf`**: PDF file containing:
  - Anonymized name (BellKor97).
  - MSE and R² values on the training set.
  - Graphs comparing the feature distribution and their correlations with the target variable.
  - Screenshot of the interaction with GPT/ChatGPT.

## Methodology
### 1. Data Preprocessing
- Handled missing values by imputing with the mean/median for numeric features.
- Normalized and scaled features as needed.

### 2. Model Training
- Utilized a **Regression Tree** algorithm to predict the target variable.
- Implemented **cross-validation** to evaluate and improve model performance, reducing the risk of overfitting.

### 3. Evaluation Metrics
- **Mean Squared Error (MSE)**: 0.0090
- **R² Score**: 0.7234

### 4. Feature Analysis
- The project includes visualizations comparing feature distributions between the training and test datasets.
- Correlation analysis between features and the target variable is also provided.

