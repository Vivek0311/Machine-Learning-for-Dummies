import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generating sample data

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

Note: here you are just join split and nothing else 
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

# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(y_train.values, label="Actual Sales", color="blue", linewidth=3)
plt.plot(y_train_pred, label="Predicted Sales", color="red", linewidth=3)
plt.title("Actual Sales vs Predicted Sales")
plt.xlabel("Index")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
