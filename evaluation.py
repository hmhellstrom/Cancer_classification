"""Module containing various helper functions used to compute evaluation metrics
for binary classification models"""

import numpy as np
from sklearn import metrics


def compute_threshold(true_labels: list, predicted: list) -> float:
    """
    Function computes optimal binary classification
    threshold using Youden's J-statistic
    """
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predicted, drop_intermediate=False)
    J_stats = tpr - fpr
    opt_thresholds = thresholds[np.argmax(J_stats)]
    return opt_thresholds


def compute_tnr(confusion_mat: np.ndarray) -> float:
    """
    A function which computes True Positive Rate using
    given confusion matrix
    """
    return confusion_mat[0, 0] / sum(confusion_mat[0, :])


def compute_tpr(confusion_mat: np.ndarray) -> float:
    """
    A function which computes True Positive Rate using
    given confusion matrix
    """
    return confusion_mat[1, 1] / sum(confusion_mat[1, :])
