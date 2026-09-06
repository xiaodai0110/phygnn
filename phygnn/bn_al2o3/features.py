# -*- coding: utf-8 -*-
"""Feature transformation and column-inference helpers for BN/Al2O3.

The raw experiment table stores physical formulation fields (``D_s``, ``D_m``,
``D_l``, ``phi_s``, ``phi_m``, ``E``). The PGNN and the Bayesian-optimization
surrogate both consume a log-transformed 6-column feature vector, so the
transform lives here as a single source of truth.
"""
import logging

import numpy as np
import pandas as pd

from phygnn.bn_al2o3.constants import LOG_FEATURE_NAMES, RAW_FEATURE_NAMES

logger = logging.getLogger(__name__)


def format_features(features, epsilon=1e-12):
    """Format raw BN/Al2O3 formulation fields into model inputs.

    Parameters
    ----------
    features : pandas.DataFrame | dict | np.ndarray
        Raw formulation table with ``D_s``, ``D_m``, ``D_l``, ``phi_s``,
        ``phi_m``, and ``E`` columns/keys, or an already formatted array.
    epsilon : float, optional
        Small positive value added before taking log(E + epsilon).

    Returns
    -------
    pandas.DataFrame | np.ndarray
        Formatted 6-column feature data:
        ``log_D_s, log_D_m, log_D_l, phi_s, phi_m, log_E``.
    """
    # 如果已经是 numpy 数组，则认为调用方已经完成了特征格式化。
    if isinstance(features, np.ndarray):
        return features

    # 允许用户传 dict，内部统一转成 DataFrame 处理列名。
    if isinstance(features, dict):
        features = pd.DataFrame(features)

    if not isinstance(features, pd.DataFrame):
        msg = ('features must be a DataFrame, dict, or already formatted '
               'ndarray, but received {}'.format(type(features)))
        logger.error(msg)
        raise TypeError(msg)

    # 如果输入已经包含标准 6 列，直接按固定顺序取出。
    if all(name in features for name in LOG_FEATURE_NAMES):
        return features.loc[:, LOG_FEATURE_NAMES].copy()

    # 否则要求输入包含文档定义的原始实验变量。
    missing = [name for name in RAW_FEATURE_NAMES if name not in features]
    if missing:
        msg = ('BN/Al2O3 features are missing required columns: {}'
               .format(missing))
        logger.error(msg)
        raise KeyError(msg)

    # 粒径和级配误差取自然对数，配比原样保留。
    out = pd.DataFrame(index=features.index)
    out['log_D_s'] = np.log(features['D_s'].astype(float))
    out['log_D_m'] = np.log(features['D_m'].astype(float))
    out['log_D_l'] = np.log(features['D_l'].astype(float))
    out['phi_s'] = features['phi_s'].astype(float)
    out['phi_m'] = features['phi_m'].astype(float)
    out['log_E'] = np.log(features['E'].astype(float) + epsilon)

    return out


def make_bo_feature_matrix(frame, epsilon=1e-12):
    """Transform raw candidate rows into the log feature space used by BO."""
    return format_features(frame, epsilon=epsilon)


def make_feature_keys(frame, feature_names=None, precision=6):
    """Create rounded feature keys for measured-candidate matching."""
    if feature_names is None:
        feature_names = RAW_FEATURE_NAMES
    rounded = frame[feature_names].round(precision)
    return rounded.astype(str).agg('|'.join, axis=1).tolist()


def infer_column(frame, candidates, role):
    """Return the first candidate column present in ``frame``."""
    for column in candidates:
        if column in frame.columns:
            return column
    msg = '{} column not found. Tried: {}'.format(role, ', '.join(candidates))
    raise ValueError(msg)
