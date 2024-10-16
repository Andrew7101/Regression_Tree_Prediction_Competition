#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:41:30 2024
@author: jeongwoohong
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import os  

# Load training and test datasets
train = pd.read_csv('/path/to/small_train_data.csv')
test = pd.read_csv('/path/to/test_data.csv')

# Convert price to its logarithmic scale to normalize the target variable
train['log_price'] = np.log(train['price'])
# Drop the original 'price' column as it's replaced by the log transformation
train.drop('price', axis=1, inplace=True)

# Define features to be used for training the model
feature_cols = [
    'city_fuel_economy',
    'engine_displacement',
    'highway_fuel_economy',
    'horsepower',
    'mileage',
    'year'
]

# Separate features (X) and target variable (y) for the training data
X = train[feature_cols]
y = train['log_price']

# Test data features
X_test = test[feature_cols]

# Model Training with Cross-Validation
# Initialize the Decision Tree Regressor with a fixed random state for reproducibility
dt_regressor = DecisionTreeRegressor(random_state=42)

# Define cross-validation strategy: 5-fold cross-validation with shuffling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define evaluation metrics: MSE and R2 score
scoring = {
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
    'R2': 'r2'
}

# Perform cross-validation and evaluate the model using the defined metrics
cv_results_initial = cross_validate(
    dt_regressor,
    X,
    y,
    scoring=scoring,
    cv=kf,
    n_jobs=-1,
    return_train_score=False
)

# Convert negative MSE scores (since sklearn returns negative values for consistency) to positive
initial_mse_scores = -cv_results_initial['test_MSE']
initial_r2_scores = cv_results_initial['test_R2']

# Output average scores for initial evaluation
print(f"Initial Average MSE: {initial_mse_scores.mean()}")
print(f"Initial Average R2: {initial_r2_scores.mean()}")
