# -*- coding: utf-8 -*-
"""Regression metrics shared by the BN/Al2O3 evaluation tooling."""
import numpy as np


def compute_regression_metrics(y_true, y_pred):
    """Calculate regression metrics from observed and predicted values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    nonzero = np.abs(y_true) > np.finfo(float).eps
    if np.any(nonzero):
        mape = np.mean(abs_error[nonzero] / np.abs(y_true[nonzero])) * 100.0
    else:
        mape = np.nan

    return {
        'n': float(len(y_true)),
        'r2': float(r2),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'mae': float(np.mean(abs_error)),
        'mape_percent': float(mape),
        'bias': float(np.mean(error)),
        'max_abs_error': float(np.max(abs_error)),
        'mean_observed': float(np.mean(y_true)),
        'mean_predicted': float(np.mean(y_pred)),
    }
