# Code written using Python 3.12.2
# -*- coding: utf-8 -*-

"""
Module Name: Regression_techniques.py
Description: This module performs regression analysis on the Boston housing dataset to predict prices. 
It includes data preprocessing, implementation of linear regression, Ridge, and Lasso regression with hyperparameter 
tuning using GridSearchCV. The model evaluations use metrics such as Mean Squared Error (MSE) and R-squared (R²).
"""

__author__ = "Vivek Sirwal"
__email__ = "viveksirwal@gmail.com"
__date__ = "2024-11-24"
__version__ = "1.0.2"
__status__ = "Production"


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

df_1 = (raw_df.iloc[0::2]).reset_index(drop=True)
df_2 = (raw_df.iloc[1::2]).reset_index(drop=True)

# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

df = (pd.concat([df_1, df_2], axis=1)).dropna(axis=1)

df.columns = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "Price",
]

X = df.iloc[:, :-1]  # independent features
y = df.iloc[:, -1]  # dependent features


#######################################
#          Linear Regression          #
#######################################

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()

# Cross validation
mse = cross_val_score(lin_reg, X, y, scoring="neg_mean_squared_error", cv=5)
"""Here parameters can be: neg_mean_squared_error, 
accuracy, mean_squared_error(not present)"""
mean_mse = np.mean(mse)
print(f"mean mse using crossvalscore using linear regression:{mean_mse}")


#######################################
#         Ridge Regression(L2)        #
#######################################

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

"""GridSearchCV is for Hyperparameter tunning,
used to find the best combination of hyperparameters for a given model"""

ridge = Ridge()
params = [{"alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]}]

# here 1st thing that goes in is my model then params
ridge_regressor = GridSearchCV(
    ridge, param_grid=params, scoring="neg_mean_squared_error", cv=5
)

ridge_regressor.fit(X, y)

print(f"best ridge parameter:{ridge_regressor.best_params_}")
print(f"best ridge score:{ridge_regressor.best_score_}")

"""1st we got -37 now because of ridge the 
mse has reduced to -32 which is good"""


#######################################
#        Lasso Regression(L1)         #
#######################################

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

"""GridSearchCV is for Hyperparameter tunning,
used to find the best combination of hyperparameters for a given model"""

lasso = Lasso()
params = [{"alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]}]

# here 1st thing that goes in is my model then params
lasso_regressor = GridSearchCV(
    lasso, param_grid=params, scoring="neg_mean_squared_error", cv=5
)

lasso_regressor.fit(X, y)

print(f"best lasso parameter:{lasso_regressor.best_params_}")
print(f"best lasso score:{lasso_regressor.best_score_}")

"""1st we got -37 now because of ridge the 
mse has reduced to -32 which is good"""


################################################
# Now adding more parameters i.e; alpha values #
################################################

params = [
    {"alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 50, 55, 100]}
]


# Ridge regression with more params
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

"""GridSearchCV is for Hyperparameter tunning,
used to find the best combination of hyperparameters for a given model"""

ridge = Ridge()
params = [
    {"alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 50, 55, 100]}
]

# here 1st thing that goes in is my model then params
ridge_regressor = GridSearchCV(
    ridge, param_grid=params, scoring="neg_mean_squared_error", cv=5
)

ridge_regressor.fit(X, y)

print(f"best ridge parameter using additional params:{ridge_regressor.best_params_}")
print(f"best ridge score using additional params:{ridge_regressor.best_score_}")

"""Now after adding more params the mse got reduced to -29
and alpha = 100"""


# Lasso regression with more params
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

"""GridSearchCV is for Hyperparameter tunning,
used to find the best combination of hyperparameters for a given model"""

lasso = Lasso()
params = [
    {"alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 50, 55, 100]}
]

# here 1st thing that goes in is my model then params
lasso_regressor = GridSearchCV(
    lasso, param_grid=params, scoring="neg_mean_squared_error"
)

lasso_regressor.fit(X, y)

print(f"best lasso parameter using additional params:{lasso_regressor.best_params_}")
print(f"best lasso score using additional params:{lasso_regressor.best_score_}")


#######################################
#       Using Train test split        #
#######################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Cross validation
mse = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
"""Here parameters can be: neg_mean_squared_error, accuracy, mean_squared_error(not present)"""
mean_mse = np.mean(mse)
print(f"Linear reg mean mse after tts:{mean_mse}")


######################################################


# Ridge regression with train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

"""GridSearchCV is for Hyperparameter tunning,
used to find the best combination of hyperparameters for a given model"""

ridge = Ridge()
params = [
    {"alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 50, 55, 100]}
]

# here 1st thing that goes in is my model then params
ridge_regressor = GridSearchCV(
    ridge, param_grid=params, scoring="neg_mean_squared_error", cv=5
)

ridge_regressor.fit(X_train, y_train)

print(f"Ridge reg best params after tts:{ridge_regressor.best_params_}")
print(f"Ridge reg best score after tts:{ridge_regressor.best_score_}")


##############################################################


# Lasso regression with train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

"""GridSearchCV is for Hyperparameter tunning,
used to find the best combination of hyperparameters for a given model"""

lasso = Lasso()
params = [
    {"alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 50, 55, 100]}
]

# here 1st thing that goes in is my model then params
lasso_regressor = GridSearchCV(
    lasso, param_grid=params, scoring="neg_mean_squared_error"
)

lasso_regressor.fit(X_train, y_train)

print(f"Ridge reg best params after tts:{lasso_regressor.best_params_}")
print(f"Ridge reg best score after tts:{lasso_regressor.best_score_}")


from sklearn.metrics import r2_score

y_pred_l = lasso_regressor.predict(X_test)
r2_score_l = r2_score(y_pred_l, y_test)
print(f"Lasso reg r square:{r2_score_l}")

y_pred_r = ridge_regressor.predict(X_test)
r2_score_r = r2_score(y_pred_r, y_test)
print(f"Ridge reg r square:{r2_score_r}")

y_pred_li = lin_reg.predict(X_test)
r2_score_1i = r2_score(y_pred_li, y_test)
print(f"Linear reg r square:{r2_score_1i}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(
    y_test.reset_index(drop=True),
    label="Actual",
    color="red",
    linewidth=3,
)
plt.plot(y_pred_l, label="Predicted", color="blue", linewidth=3)
plt.title("(Lasso regression) Actual vs Predicted")
plt.grid(True)
plt.legend()
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(
    y_test.reset_index(drop=True),
    label="Actual",
    color="red",
    linewidth=3,
)
plt.plot(y_pred_r, label="Predicted", color="blue", linewidth=3)
plt.title("(Ridge regression) Actual vs Predicted")
plt.grid(True)
plt.legend()
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(
    y_test.reset_index(drop=True),
    label="Actual",
    color="red",
    linewidth=3,
)
plt.plot(y_pred_li, label="Predicted", color="blue", linewidth=3)
plt.title("(Linear regression) Actual vs Predicted")
plt.grid(True)
plt.legend()
plt.show()
plt.close()
