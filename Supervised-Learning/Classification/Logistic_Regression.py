#################################################
#              Logistic Regression              #
#################################################

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


df = load_breast_cancer()
X = pd.DataFrame(df["data"], columns=df["feature_names"])  # Independent features
y = pd.DataFrame(df["target"], columns=["Target"])  # Dependent features
# just to check balanced data or imbalanced data
# (biased or high varient data)
y["Target"].value_counts()

####################
# Train test split #
####################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

params = [{"C": [1, 5, 10], "max_iter": [100, 150]}]

model_1 = LogisticRegression("l2", C=100, max_iter=100)

model = GridSearchCV(model_1, param_grid=params, scoring="f1", cv=5)

model.fit(X_train, y_train)

print(f"Logistic reg using tts gridsearchcv best params:{model.best_params_}")
print(f"Logistic reg using tts gridsearchcv best score:{model.best_score_}")


y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

confusion_matrix(y_pred, y_test)
accuracy_score(y_pred, y_test)

# Detailed performance metrics
print(classification_report(y_pred, y_test))
