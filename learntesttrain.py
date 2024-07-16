import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_pred = pd.DataFrame(
    {
        "Tactic": ["A", "B", "C"],
        "Measure": [7.2, 5.5, 9.1],
    }
)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the linear regression model
model = LinearRegression()
