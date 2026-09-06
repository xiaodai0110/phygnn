# -*- coding: utf-8 -*-
"""Shared column names and constants for the BN/Al2O3 PGNN workflow.

This module is the single source of truth for feature/label names so the
scripts, the model interface, and the Bayesian-optimization surrogate all
agree on the same schema instead of re-declaring it inline.
"""

# 文档中第一版 BN/Al2O3 方案固定的 6 个原始实验输入列。
RAW_FEATURE_NAMES = ['D_s', 'D_m', 'D_l', 'phi_s', 'phi_m', 'E']

# 神经网络实际消费的对数变换后的 6 列。
LOG_FEATURE_NAMES = [
    'log_D_s',
    'log_D_m',
    'log_D_l',
    'phi_s',
    'phi_m',
    'log_E',
]

# 监督标签统一命名为总热导率 k。
LABEL_NAMES = ['k']

# 自动探测“真实观测热导率”列时的候选顺序。
OBSERVED_K_CANDIDATES = [
    'k_HF',
    'k',
    'k_LF',
    'thermal_conductivity',
    'conductivity',
    'measured_k',
]

# 自动探测“预测热导率”列时的候选顺序。
PREDICTED_K_CANDIDATES = ['k_pred', 'predicted_k', 'prediction']

# 分组/来源列候选，用于画图着色或训练集划分。
GROUP_COLUMN_CANDIDATES = ['lf_split', 'split', 'source']

# 原始测量表与配方表合并时尝试的共享 ID 列。
MERGE_KEY_CANDIDATES = ['experiment_id', 'sample_id', 'formulation_id']

# 高保真整理表中需要保留透传的元数据列。
METADATA_COLUMNS = [
    'formulation_id',
    'triplet_id',
    'small_id',
    'medium_id',
    'large_id',
    'D_s',
    'D_m',
    'D_l',
    'phi_s',
    'phi_m',
    'phi_l',
    'E',
    'V_f',
    'rank',
    'k_pred',
]

# 配方生成时透传的可选候选/预测列。
OPTIONAL_INPUT_COLUMNS = [
    'triplet_id',
    'small_id',
    'medium_id',
    'large_id',
    'rank',
    'k_pred',
    'Delta_k_pred',
    'k',
    'k_LF',
]
