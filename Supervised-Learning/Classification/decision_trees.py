import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

"""Here max_depth, min_samples_leaf, max_features are parameters"""
classifier = DecisionTreeClassifier()

classifier.fit(iris.data, iris.target)

plt.figure(figsize=(15, 10))
tree.plot_tree(classifier, filled=True)
