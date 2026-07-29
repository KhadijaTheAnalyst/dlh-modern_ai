#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
Logistic_Regression_Model = __import__('5-logisitc_regression').Logistic_Regression_Model

seed = 42
np.random.seed(seed)

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    random_state=seed
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

model = Logistic_Regression_Model(random_state=seed)
model.fit(X_train, y_train)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]
prob = model.predict_proba(grid)[:, 1]
prob = prob.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, prob, levels=50, cmap='viridis', alpha=0.6)
plt.colorbar(label='Predicted Probability')
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='purple', edgecolor='k', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='orange', edgecolor='k', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Predicted Probability Contour')
plt.legend()
plt.savefig("5_main_2.png")
plt.show()
