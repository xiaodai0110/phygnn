# -*- coding: utf-8 -*-
"""BN/Al2O3 multi-fidelity PGNN domain helpers.

This subpackage centralizes the constants, feature transforms, metrics,
physics, and data pipeline shared by the :mod:`workflow` scripts, so the
scripts no longer import from one another.
"""
from phygnn.bn_al2o3.constants import (
    GROUP_COLUMN_CANDIDATES,
    LABEL_NAMES,
    LOG_FEATURE_NAMES,
    MERGE_KEY_CANDIDATES,
    METADATA_COLUMNS,
    OBSERVED_K_CANDIDATES,
    OPTIONAL_INPUT_COLUMNS,
    PREDICTED_K_CANDIDATES,
    RAW_FEATURE_NAMES,
)
from phygnn.bn_al2o3.data import (
    demo_powder_table,
    generate_lf_data,
    load_df_candidates,
    load_powder_table,
    merge_packing_data,
    normalize_df_candidates,
    normalize_powder_table,
    save_training_tables,
    select_df_triplets,
    write_template,
)
from phygnn.bn_al2o3.features import (
    format_features,
    infer_column,
    make_bo_feature_matrix,
    make_feature_keys,
)
from phygnn.bn_al2o3.metrics import compute_regression_metrics
from phygnn.bn_al2o3.paths import (
    data_path,
    models_path,
    outputs_path,
    repo_root,
)
from phygnn.bn_al2o3.physics import (
    dinger_funk_target,
    estimate_phi_max_from_error,
    lewis_nielsen_conductivity,
)

__all__ = [
    'GROUP_COLUMN_CANDIDATES',
    'LABEL_NAMES',
    'LOG_FEATURE_NAMES',
    'MERGE_KEY_CANDIDATES',
    'METADATA_COLUMNS',
    'OBSERVED_K_CANDIDATES',
    'OPTIONAL_INPUT_COLUMNS',
    'PREDICTED_K_CANDIDATES',
    'RAW_FEATURE_NAMES',
    'compute_regression_metrics',
    'data_path',
    'demo_powder_table',
    'dinger_funk_target',
    'estimate_phi_max_from_error',
    'format_features',
    'generate_lf_data',
    'infer_column',
    'lewis_nielsen_conductivity',
    'load_df_candidates',
    'load_powder_table',
    'make_bo_feature_matrix',
    'make_feature_keys',
    'merge_packing_data',
    'models_path',
    'normalize_df_candidates',
    'normalize_powder_table',
    'outputs_path',
    'repo_root',
    'save_training_tables',
    'select_df_triplets',
    'write_template',
]
