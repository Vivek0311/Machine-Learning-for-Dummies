# Code written using Python 3.12.2
# -*- coding: utf-8 -*-

"""
Module Name: Linear_Regression.py
Description: This module implements a linear regression analysis to predict sales based on synthetic data generated over 100 days. 
It creates a dataset with random tactics and measures, splits the data into training and testing sets, and trains a linear regression 
model using the "Measure" feature. The model's performance is evaluated using metrics such as Mean Squared Error (MSE), 
Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R-squared (R²). 
Visualization plots are included to compare actual and predicted sales, along with error analysis for each prediction.
"""

__author__ = "Vivek Sirwal"
__email__ = "viveksirwal@gmail.com"
__date__ = "2024-10-28"
__version__ = "1.0.5"
__status__ = "Production"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

"""Generating sample data"""
# the random seed is set to 0 as each time I generate random number it remains the same
np.random.seed(0)
days = np.arange(1, 101)  # 100 days of data

tactics = np.random.choice(["A", "B", "C"], size=100)  # Three tactics A, B, C

measures = np.random.rand(100) * 10  # Random measures as independent variable
sales = 50 + 5 * measures + np.random.randn(100) * 10  # Sales as dependent variable

data = pd.DataFrame(
    {"Day": days, "Tactic": tactics, "Measure": measures, "Sales": sales}
)


# Splitting data into X and y
X = data[["Tactic", "Measure"]]  # Features
y = data["Sales"]  # Target variable

"""here we are doing test train split
parameters *array(X, y is array (Allowed inputs are lists, numpy arrays, scipy-sparse
matrices or pandas dataframes.)

here you have stratify     
        stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels."""

"""refers to a method of sampling data (usually during training or validation) 
where the data is divided into different strata or groups. 
These groups typically represent different classes or categories within the data.

Note: here you are just splitting the data into two datasets and nothing else 
after this you need to train the model based on your priority 
like linear, ridge, lasso, decision trees etc"""

# In the context of machine learning algorithms,
# random_state is a parameter that allows you to specify a seed value
# for the random number generator used in certain operations.
"""Here we are splitting the set into two parts training set(80%) and testing set(20%)
The logic is we train using the 80% data and test it on remaining 20%"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the linear regression model
model = LinearRegression()

"""Fitting the model using training set"""
model.fit(X_train[["Measure"]], y_train)

# Using the training set(X-Measure) I'm predicting the Y-Sales
y_train_pred = model.predict(X_train[["Measure"]])

# The equation is Y = βX + C
# Sales = Betacoefficeints(Measure) + intercept
# you can extract the intercept and coefficients like done below
y_intercept = model.intercept_
coefficient = model.coef_

"""After fitting and predicting we use something called as performance metrics
To determine how good is our model behaving with new data(test data / new inputs)
Note: This step is imp for evaluating model Below are some of performance metrics used to evaluate
"""
"""1. Mean Squared Error (MSE): Measures the average squared difference between
        predicted and true values.
        Formula: MSE = (1/n) * Σ(y_train - y_train_pred)^2
        Indicator: Measures the average squared difference between predicted and actual values. 
        It penalizes larger errors more heavily than smaller errors.
        When to Use: Best used when larger errors should be penalized more severely, 
        such as in scenarios where large deviations are costly or indicative of severe misprediction."""

mse = mean_squared_error(y_train, y_train_pred)

"""2. Root Mean Squared Error (RMSE): Measures the square root of the average squared difference between
        predicted and true values. It is more interpretable than MSE as it is in the same unit as the target.
        Formula: RMSE = sqrt(MSE)
        Indicator: Provides the square root of MSE, making it more interpretable as it is in the same unit 
        as the target variable. It emphasizes large errors, like MSE.
        When to Use: Use RMSE when interpretability in the same units as the target variable is crucial, 
        making it easier to contextualize model performance."""
rmse = np.sqrt(mse)

"""3. Mean Absolute Error (MAE): Measures the average absolute difference between predicted and true values,
        showing how much, on average, the predictions deviate from the actual values.
        Formula: MAE = (1/n) * Σ|y_train - y_train_pred|
        Indicator: Calculates the average absolute difference between predictions and actual values, 
        giving an easily interpretable measure of error magnitude.
        When to Use: Use MAE when all prediction errors should be treated equally and a simple, 
        interpretable metric is needed without penalizing larger errors disproportionately."""
mae = mean_absolute_error(y_train, y_train_pred)

"""4. Mean Absolute Percentage Error (MAPE): Measures the average absolute percentage difference between predicted
        and true values, showing the error as a percentage.
        Formula: MAPE = (1/n) * Σ(|(y_train - y_train_pred) / y_train|) * 100
        Indicator: Expresses the error as a percentage of the actual values, 
        making it easier to interpret in relative terms rather than absolute.
        When to Use: Best used when you want to understand the error relative to the magnitude of 
        actual values, especially in cases where understanding relative deviations is important."""
mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100

"""5. R-squared (R2): Indicates the proportion of variance in the target variable explained by the model. 
        Value range is [0,1] where 1 indicates a perfect model.
        Formula: R2 = 1 - (Σ(y_train - y_train_pred)^2 / Σ(y_train - y_train_mean)^2)
        Indicator: How well the model fits the data
        Indicator: Reflects the proportion of variance in the target variable explained by the model, with a value range of [0, 1]. 
        Higher values indicate better model fit.
        When to Use: Use R² to understand how well the model captures the variance in the data. 
        Ideal for model evaluation when comparing multiple models, as it provides a quick sense of goodness-of-fit."""
r_squared = model.score(X_train[["Measure"]], y_train)

# Plotting graph for understanding
# the below graph is between Actual sales(that we used for training set)
# vs the one which model predicted

# scatter plot (uncomment if you wanna use this)
# plt.figure(figsize=(10, 6))
# plt.scatter(X_train["Measure"], y_train, color="blue", label="Actual")
# plt.plot(X_train["Measure"], y_train_pred, color="red", linewidth=2, label="Predicted")
# plt.title("Linear Regression: Actual vs Predicted Sales")
# plt.xlabel("Measure")
# plt.ylabel("Sales")
# plt.legend()
# plt.grid(True)
# plt.show()


print(f"Y_Intercept, coefficients are {y_intercept, coefficient }")

# predicted graph
plt.figure(figsize=(10, 6))
plt.plot(y_train_pred, label="Predicted Sales", color="red")
plt.title("Predicted Sales")
plt.xlabel("Index")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# Actual graph
plt.figure(figsize=(10, 6))
plt.plot(y_train.values, label="Actual Sales", color="blue")
plt.title("Actual Sales")
plt.xlabel("Index")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted for training data
plt.figure(figsize=(10, 6))
plt.plot(y_train.values, label="Actual Sales", color="blue", linewidth=3)
plt.plot(y_train_pred, label="Predicted Sales", color="red", linewidth=3)
plt.title("Actual Sales vs Predicted Sales")
plt.xlabel("Index")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# Trying it for test data
X_test_pred = model.predict(X_test[["Measure"]])

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Sales", color="blue", linewidth=3)
plt.plot(X_test_pred, label="Predicted Sales", color="red", linewidth=3)
plt.title("Actual Sales vs Predicted Sales")
plt.xlabel("Index")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# Check the prediction and error on complete dataset
data["Predicted"] = model.predict(data[["Measure"]])
data["Error"] = data["Sales"] - data["Predicted"]
