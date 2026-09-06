"""Suggest the next BN/Al2O3 experiments from a trained PGNN model.

This script ranks candidate formulations for the next experimental batch.
It can either:

1. Load a saved BN/Al2O3 PGNN model and predict k.
2. Use an existing k_pred column if no model is available.

The recommender is intentionally simple and auditable: score candidates by
predicted conductivity, optional grading-error penalty, optional exploration
distance from existing HF data, and a per-triplet diversity cap.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd

from phygnn.bn_al2o3.constants import RAW_FEATURE_NAMES
from phygnn.bn_al2o3.features import make_feature_keys
from phygnn.bn_al2o3.paths import models_path, outputs_path
from phygnn.model_interfaces.phygnn_model import PhygnnModel


def build_parser():
    """Build command-line arguments for experiment suggestion."""
    parser = argparse.ArgumentParser(
        description='Suggest next BN/Al2O3 experiments from PGNN candidates.'
    )
    parser.add_argument(
        '--candidates',
        type=Path,
        default=outputs_path('bn_al2o3_predictions.csv'),
        help='Candidate CSV with PGNN feature columns.',
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=models_path('bn_al2o3_model'),
        help='Saved PhygnnModel directory/json/pkl.',
    )
    parser.add_argument(
        '--hf-data',
        type=Path,
        default=None,
        help='Optional measured HF table for exclusion/exploration.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=outputs_path('next_experiment_suggestions.csv'),
    )
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--max-per-triplet', type=int, default=2)
    parser.add_argument('--error-weight', type=float, default=0.0)
    parser.add_argument('--exploration-weight', type=float, default=0.0)
    parser.add_argument('--max-e', type=float, default=None)
    parser.add_argument('--min-k-pred', type=float, default=None)
    parser.add_argument('--key-precision', type=int, default=6)
    parser.add_argument('--no-model', action='store_true')
    parser.add_argument('--no-exclude-measured', action='store_true')
    return parser


def load_candidate_table(path):
    """Load and validate candidate rows."""
    if not path.exists():
        msg = 'Candidate file not found: {}'.format(path)
        raise FileNotFoundError(msg)

    data = pd.read_csv(path)
    missing = [col for col in RAW_FEATURE_NAMES if col not in data.columns]
    if missing:
        msg = 'Candidate table is missing columns: {}'.format(
            ', '.join(missing)
        )
        raise ValueError(msg)

    for col in RAW_FEATURE_NAMES:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    invalid = data[RAW_FEATURE_NAMES].isna().any(axis=1)
    if invalid.any():
        bad_rows = ', '.join(map(str, data.index[invalid].tolist()[:10]))
        msg = 'Candidate table has invalid rows: {}'.format(bad_rows)
        raise ValueError(msg)

    if 'phi_l' not in data.columns and {'phi_s', 'phi_m'} <= set(data):
        data['phi_l'] = 1.0 - data['phi_s'] - data['phi_m']

    return data.reset_index(drop=True)


def load_hf_table(path):
    """Load optional measured HF data."""
    if path is None:
        return None
    if not path.exists():
        msg = 'HF data file not found: {}'.format(path)
        raise FileNotFoundError(msg)

    data = pd.read_csv(path)
    missing = [col for col in RAW_FEATURE_NAMES if col not in data.columns]
    if missing:
        msg = 'HF data is missing columns: {}'.format(', '.join(missing))
        raise ValueError(msg)
    for col in RAW_FEATURE_NAMES:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data.dropna(subset=RAW_FEATURE_NAMES).reset_index(drop=True)


def load_model(path, no_model=False):
    """Load a saved PhygnnModel if requested and available."""
    if no_model:
        return None
    if path is None or not path.exists():
        return None
    return PhygnnModel.load(str(path))


def score_candidates(
    candidates,
    model=None,
    hf_data=None,
    error_weight=0.0,
    exploration_weight=0.0,
):
    """Add k predictions and recommendation score to candidates."""
    scored = candidates.copy()
    features = scored[RAW_FEATURE_NAMES].copy()

    if model is not None:
        scored['k_pred'] = model.predict_k(
            features,
            table=True,
        ).iloc[:, 0].to_numpy()
        scored['Delta_k_pred'] = model.predict_delta(
            features,
            table=True,
        ).iloc[:, 0].to_numpy()
        scored['prediction_source'] = 'loaded_model'
    elif 'k_pred' in scored.columns:
        scored['k_pred'] = pd.to_numeric(scored['k_pred'], errors='coerce')
        scored['prediction_source'] = 'existing_k_pred'
    else:
        msg = 'No model found and candidates do not contain k_pred.'
        raise ValueError(msg)

    scored = scored.dropna(subset=['k_pred']).reset_index(drop=True)
    scored['grading_error_penalty'] = error_weight * scored['E']
    scored['nearest_hf_distance'] = 0.0
    if exploration_weight > 0 and hf_data is not None and not hf_data.empty:
        scored['nearest_hf_distance'] = nearest_hf_distance(scored, hf_data)

    scored['recommendation_score'] = (
        scored['k_pred']
        - scored['grading_error_penalty']
        + exploration_weight * scored['nearest_hf_distance']
    )
    scored['recommendation_reason'] = build_reason(scored, exploration_weight)
    return scored


def filter_candidates(
    candidates,
    *,
    hf_data=None,
    max_e=None,
    min_k_pred=None,
    exclude_measured=True,
    key_precision=6,
):
    """Apply practical experiment filters before ranking."""
    filtered = candidates.copy()
    if max_e is not None:
        filtered = filtered.loc[filtered['E'] <= max_e]
    if min_k_pred is not None:
        filtered = filtered.loc[filtered['k_pred'] >= min_k_pred]

    if exclude_measured and hf_data is not None and not hf_data.empty:
        measured_keys = set(make_feature_keys(hf_data, key_precision))
        candidate_keys = make_feature_keys(filtered, key_precision)
        keep = [key not in measured_keys for key in candidate_keys]
        filtered = filtered.loc[keep]

    return filtered.reset_index(drop=True)


def select_diverse_top(candidates, top_n=5, max_per_triplet=2):
    """Select top candidates while limiting repeated triplets."""
    if top_n <= 0:
        msg = 'top_n must be greater than zero.'
        raise ValueError(msg)
    if max_per_triplet <= 0:
        msg = 'max_per_triplet must be greater than zero.'
        raise ValueError(msg)

    ranked = candidates.sort_values(
        ['recommendation_score', 'k_pred'],
        ascending=[False, False],
    )
    selected = []
    triplet_counts = {}

    for _, row in ranked.iterrows():
        triplet = row.get('triplet_id', 'unknown_triplet')
        count = triplet_counts.get(triplet, 0)
        if count >= max_per_triplet:
            continue

        selected.append(row)
        triplet_counts[triplet] = count + 1
        if len(selected) >= top_n:
            break

    if not selected:
        return ranked.head(0).copy()

    output = pd.DataFrame(selected).reset_index(drop=True)
    output.insert(0, 'recommendation_rank', np.arange(1, len(output) + 1))
    return _order_suggestion_columns(output)


def nearest_hf_distance(candidates, hf_data):
    """Calculate distance from each candidate to the nearest HF point."""
    x_candidates = candidates[RAW_FEATURE_NAMES].to_numpy(dtype=float)
    x_hf = hf_data[RAW_FEATURE_NAMES].to_numpy(dtype=float)
    stacked = np.vstack([x_candidates, x_hf])
    scale = stacked.std(axis=0)
    scale[scale == 0] = 1.0

    diff = (
        x_candidates[:, None, :] - x_hf[None, :, :]
    ) / scale[None, None, :]
    return np.sqrt(np.sum(diff**2, axis=2)).min(axis=1)


def build_reason(scored, exploration_weight):
    """Build short auditable recommendation reason strings."""
    reasons = []
    for _, row in scored.iterrows():
        parts = ['high predicted k']
        if row['E'] <= scored['E'].quantile(0.25):
            parts.append('low DF error')
        if exploration_weight > 0 and row['nearest_hf_distance'] > 0:
            parts.append('adds feature-space exploration')
        reasons.append('; '.join(parts))
    return reasons


def main(argv=None):
    """Run the recommendation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    candidates = load_candidate_table(args.candidates)
    hf_data = load_hf_table(args.hf_data)
    model = load_model(args.model, no_model=args.no_model)
    if model is None:
        print('No model loaded; using existing k_pred column if available.')

    scored = score_candidates(
        candidates,
        model=model,
        hf_data=hf_data,
        error_weight=args.error_weight,
        exploration_weight=args.exploration_weight,
    )
    filtered = filter_candidates(
        scored,
        hf_data=hf_data,
        max_e=args.max_e,
        min_k_pred=args.min_k_pred,
        exclude_measured=not args.no_exclude_measured,
        key_precision=args.key_precision,
    )
    suggestions = select_diverse_top(
        filtered,
        top_n=args.top_n,
        max_per_triplet=args.max_per_triplet,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suggestions.to_csv(args.output, index=False, encoding='utf-8-sig')

    preview_cols = [
        'recommendation_rank',
        'triplet_id',
        'D_s',
        'D_m',
        'D_l',
        'phi_s',
        'phi_m',
        'E',
        'k_pred',
        'recommendation_score',
    ]
    preview_cols = [col for col in preview_cols if col in suggestions.columns]
    print('Saved {} suggestions to {}'.format(len(suggestions),
                                              args.output.resolve()))
    print(suggestions[preview_cols].head(args.top_n).to_string(index=False))
    return 0


def _order_suggestion_columns(output):
    first_cols = [
        'recommendation_rank',
        'recommendation_score',
        'recommendation_reason',
        'prediction_source',
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
        'k_pred',
        'Delta_k_pred',
        'grading_error_penalty',
        'nearest_hf_distance',
    ]
    ordered = [col for col in first_cols if col in output.columns]
    remaining = [col for col in output.columns if col not in ordered]
    return output[ordered + remaining]


if __name__ == '__main__':
    raise SystemExit(main())
