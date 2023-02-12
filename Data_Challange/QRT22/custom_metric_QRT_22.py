"""
Custom metric for QRT 2022.
"""

import numpy as np
import pandas as pd


def transform_submission_to_ypred(df_A_beta, x_test, y_test):
    """ Transform submission output (A, beta) into predicted returns S_t."""
    df_A_beta = df_A_beta.to_numpy()
    
    A = df_A_beta[:-10].reshape((250, 10))
    beta = df_A_beta[-10:].reshape(10)

    E = pd.DataFrame(A.T @ A - np.eye(10)).abs()  

    # check orthogonality of A
    if any(E.unstack() > 1e-6): 
        return None

    x_test = x_test.T
    y_test = y_test.T

    x_test = x_test[y_test.columns]

    x_test_reshape = pd.concat([x_test.shift(i+1).stack(dropna=False) for i in range(250)], 1).dropna()
    y_pred = (x_test_reshape @ A @ beta).unstack()

    return y_pred.T


def metric(df_y_true, df_y_pred):
    """ Compute metric. """
    if df_y_pred is None:  # If the y_pred has only zeroes, the metric is set to -1.
        return -1.0
    
    y_true = df_y_true.T
    y_pred = df_y_pred.T
    
    y_true = y_true.div(y_true.pow(2.0).sum(1).pow(0.5), 0)
    y_pred = y_pred.div(y_pred.pow(2.0).sum(1).pow(0.5), 0)

    mean_overlap = (y_true * y_pred).sum(1).mean()

    return mean_overlap
