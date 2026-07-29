#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
Linear_Regression = __import__('0-linear_regression').Linear_Regression
ridge_regression = __import__('2-ridge_regression').ridge_regression
get_shap_explainer_and_values = __import__('4-shap').get_shap_explainer_and_values


np.random.seed(42)
feature_names = ["X1", "X2", "X3"]

X1 = np.random.rand(200) * 10
X2 = X1 + np.random.normal(0, 0.05, 200)
X3 = np.random.rand(200) * 5
X = np.column_stack([X1, X2, X3])
y = 4*X1 + 3*X3 + np.random.normal(0, 5, 200)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

lr = Linear_Regression()
lr.fit(X_train, y_train)

ridge = ridge_regression(random_state=42)
ridge.fit(X_train, y_train)

explainer_lr, shap_values_lr = get_shap_explainer_and_values(lr, X_train, X_test)
explainer_ridge, shap_values_ridge = get_shap_explainer_and_values(ridge, X_train, X_test)

shap_values_lr.feature_names = feature_names
shap_values_ridge.feature_names = feature_names

plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
shap.plots.bar(shap_values_lr, show=False)
plt.title("SHAP Feature Importance – Linear Regression")
plt.gca().set_xlabel("Mean Absolute SHAP Values Across Samples")

plt.subplot(2, 2, 2)
shap.plots.bar(shap_values_ridge, show=False)
plt.title("SHAP Feature Importance – Ridge Regression")
plt.gca().set_xlabel("Mean Absolute SHAP Values Across Samples")

plt.subplot(2, 2, 3)
shap.plots.beeswarm(shap_values_lr, show=False)
plt.title("SHAP Beeswarm Plot – Linear Regression")

plt.subplot(2, 2, 4)
shap.plots.beeswarm(shap_values_ridge, show=False)
plt.title("SHAP Beeswarm Plot – Ridge Regression")

plt.tight_layout()
plt.savefig("4_main_1.png")
plt.show()
