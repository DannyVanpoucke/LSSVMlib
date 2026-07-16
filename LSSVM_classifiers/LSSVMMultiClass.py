#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

"""
Least Squares Support Vector Machine (LS-SVM)
Multi-class Classification

This implementation extends the binary LS-SVM classifier
using a One-vs-Rest (OvR) strategy.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from LSSVMClassification import LSSVMClassification


class LSSVMMultiClass(BaseEstimator, ClassifierMixin):
    """
    Multi-class Least Squares Support Vector Machine classifier.

    This implementation trains one binary LS-SVM classifier
    for every class using a One-vs-Rest strategy.
    """

    def __init__(self,
                 gamma: float = 1.0,
                 kernel: str = None,
                 c: float = 1.0,
                 d: float = 2,
                 sigma: float = 1.0):
        """
        Create a new multi-class LS-SVM classifier.

        Parameters
        ----------
        gamma : float
            Regularization parameter.

        kernel : str
            Kernel type ('linear', 'poly', 'rbf').

        c : float
            Polynomial kernel scaling constant.

        d : float
            Polynomial kernel degree.

        sigma : float
            RBF kernel width.
        """

        self.gamma = gamma
        self.kernel = kernel
        self.c = c
        self.d = d
        self.sigma = sigma

        # Dictionary that stores one binary classifier for every class
        self.models = {}

        # Original class labels
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train one binary LS-SVM classifier for every class
        using a One-vs-Rest strategy.

        Parameters
        ----------
        X : ndarray
            Training feature matrix.

        y : ndarray
            Target labels.

        Returns
        -------
        self
            Trained classifier.
        """

        # Store original class labels
        self.classes = np.unique(y)

        # Remove previously trained models
        self.models = {}

        # Train one binary classifier for every class
        for label in self.classes:

            # Current class -> +1
            # Remaining classes -> -1
            y_binary = np.where(y == label, 1, -1)

            model = LSSVMClassification(
                gamma=self.gamma,
                kernel=self.kernel,
                c=self.c,
                d=self.d,
                sigma=self.sigma
            )

            model.fit(X, y_binary)

            self.models[label] = model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for unseen samples.

        Parameters
        ----------
        X : ndarray
            Feature vectors to classify.

        Returns
        -------
        ndarray
            Predicted class labels.
        """

        # Collect the decision values of every
        # binary One-vs-Rest classifier
        scores = []

        for label in self.classes:

            score = self.models[label].decision_function(X)

            scores.append(score)

        # Rows = samples
        # Columns = classifiers
        scores = np.column_stack(scores)

        # Choose classifier with highest decision value
        best = np.argmax(scores, axis=1)

        # Convert indices back to original class labels
        predictions = self.classes[best]

        return predictions





