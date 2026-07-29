#!/usr/bin/env python3
"""Defines a function that creates an SVM classifier with a given kernel."""
from sklearn import svm


def get_SVM_model(name, random_state):
    """
    Creates a Support Vector Machine (SVM) classifier with the
    specified kernel using scikit-learn.

    Args:
        name: a string indicating the type of kernel to use.
            Accepted values are:
                'linear': SVM model with a linear kernel.
                'poly': SVM model with a polynomial kernel.
                'rbf': SVM model with a radial basis function
                    (RBF) kernel.
        random_state: the seed used by the random number generator
            for reproducibility.

    Returns:
        model: an untrained SVC instance with the specified kernel.
    """
    model = svm.SVC(kernel=name, random_state=random_state)
    return model
