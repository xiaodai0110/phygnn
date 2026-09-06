"""Bayesian optimization screening for BN/Al2O3 PGNN predictions.

This script is used after ``train_bn_al2o3.py`` has produced
``bn_al2o3_predictions.csv``. The default workflow is:

1. Take the top 10 candidates ranked by PGNN ``k_pred``.
2. Fit a Gaussian Process surrogate.
3. Calculate Expected Improvement (EI).
4. Output the best BO-ranked candidates for experiments.

If high-fidelity data is supplied, the GP is trained on measured ``k_HF``.
Otherwise, the GP is trained on PGNN predictions as pseudo-observations. That
fallback is useful for screening, but it is not a substitute for true
experiment-feedback Bayesian optimization.
"""

from __future__ import annotations

import argparse
import os
import warnings
from math import erf, pi, sqrt
from pathlib import Path

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.preprocessing import StandardScaler

from phygnn.bn_al2o3.constants import OBSERVED_K_CANDIDATES, RAW_FEATURE_NAMES
from phygnn.bn_al2o3.features import (
    infer_column,
    make_bo_feature_matrix,
    make_feature_keys,
)
from phygnn.bn_al2o3.paths import models_path, outputs_path
from phygnn.model_interfaces.phygnn_model import PhygnnModel

SQRT_TWO = sqrt(2.0)
SQRT_TWO_PI = sqrt(2.0 * pi)


def build_parser():
    """Build command-line arguments for BO screening."""
    parser = argparse.ArgumentParser(
        description='Run GP+EI Bayesian screening on BN/Al2O3 candidates.'
    )
    parser.add_argument(
        '--candidates',
        type=Path,
        default=outputs_path('bn_al2o3_predictions.csv'),
        help='Candidate table from train_bn_al2o3.py.',
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=models_path('bn_al2o3_model'),
        help='Saved PhygnnModel directory/json/pkl.',
    )
    parser.add_argument('--hf-data', type=Path, default=None)
    parser.add_argument('--hf-label-col', default=None)
    parser.add_argument('--hf-split-col', default='split')
    parser.add_argument('--exclude-hf-splits', default='blind')
    parser.add_argument(
        '--output',
        type=Path,
        default=outputs_path('bo_experiment_suggestions.csv'),
    )
    parser.add_argument('--pgnn-top-n', type=int, default=10)
    parser.add_argument('--bo-top-n', type=int, default=5)
    parser.add_argument('--candidate-pool', choices=['top-pgnn', 'all'],
                        default='all')
    parser.add_argument(
        '--gp-train-source',
        choices=['auto', 'hf', 'pgnn_top', 'pgnn_all'],
        default='auto',
    )
    parser.add_argument(
        '--acquisition',
        choices=['ei', 'ucb', 'hybrid'],
        default='hybrid',
    )
    parser.add_argument('--xi', type=float, default=0.01)
    parser.add_argument('--ucb-kappa', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=1e-6)
    parser.add_argument('--n-restarts-optimizer', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-per-triplet', type=int, default=2)
    parser.add_argument('--error-weight', type=float, default=0.0)
    parser.add_argument('--pgnn-score-weight', type=float, default=0.0)
    parser.add_argument('--key-precision', type=int, default=6)
    parser.add_argument('--refresh-pgnn-predictions', action='store_true')
    parser.add_argument('--no-model', action='store_true')
    parser.add_argument('--no-exclude-measured', action='store_true')
    return parser


def load_candidates(path):
    """Load candidate rows and validate PGNN feature columns."""
    if not path.exists():
        msg = 'Candidate file not found: {}'.format(path)
        raise FileNotFoundError(msg)

    candidates = pd.read_csv(path)
    missing = [
        col for col in RAW_FEATURE_NAMES if col not in candidates.columns
    ]
    if missing:
        msg = 'Candidate table is missing columns: {}'.format(
            ', '.join(missing)
        )
        raise ValueError(msg)

    for col in RAW_FEATURE_NAMES:
        candidates[col] = pd.to_numeric(candidates[col], errors='coerce')

    invalid = (
        candidates[RAW_FEATURE_NAMES].isna().any(axis=1)
        | (candidates[['D_s', 'D_m', 'D_l']] <= 0).any(axis=1)
        | (candidates[['phi_s', 'phi_m']] < 0).any(axis=1)
        | (candidates['E'] < 0)
    )
    if invalid.any():
        bad_rows = ', '.join(map(str, candidates.index[invalid].tolist()[:10]))
        msg = 'Candidate table has invalid rows: {}'.format(bad_rows)
        raise ValueError(msg)

    if 'phi_l' not in candidates.columns:
        candidates['phi_l'] = 1.0 - candidates['phi_s'] - candidates['phi_m']

    return candidates.reset_index(drop=True)


def maybe_load_model(path, no_model=False):
    """Load a saved PGNN model if available."""
    if no_model:
        return None
    if path is None or not path.exists():
        return None
    return PhygnnModel.load(str(path))


def ensure_pgnn_predictions(candidates, model=None, refresh=False):
    """Ensure candidate rows contain a numeric k_pred column."""
    output = candidates.copy()
    need_prediction = refresh or 'k_pred' not in output.columns

    if need_prediction and model is None:
        msg = 'Need a model to create k_pred, or provide existing k_pred.'
        raise ValueError(msg)

    if need_prediction:
        features = output[RAW_FEATURE_NAMES].copy()
        output['k_pred'] = model.predict_k(
            features,
            table=True,
        ).iloc[:, 0].to_numpy()
        output['Delta_k_pred'] = model.predict_delta(
            features,
            table=True,
        ).iloc[:, 0].to_numpy()

    output['k_pred'] = pd.to_numeric(output['k_pred'], errors='coerce')
    output = output.dropna(subset=['k_pred']).reset_index(drop=True)
    return output


def load_hf_data(path, label_col=None):
    """Load optional measured HF data."""
    if path is None:
        return None
    if not path.exists():
        msg = 'HF data file not found: {}'.format(path)
        raise FileNotFoundError(msg)

    hf_data = pd.read_csv(path)
    missing = [col for col in RAW_FEATURE_NAMES if col not in hf_data]
    if missing:
        msg = 'HF data is missing columns: {}'.format(', '.join(missing))
        raise ValueError(msg)

    target_col = label_col or infer_column(
        hf_data,
        OBSERVED_K_CANDIDATES,
        'measured conductivity',
    )
    for col in RAW_FEATURE_NAMES:
        hf_data[col] = pd.to_numeric(hf_data[col], errors='coerce')
    hf_data['bo_target'] = pd.to_numeric(hf_data[target_col], errors='coerce')
    return hf_data.dropna(
        subset=RAW_FEATURE_NAMES + ['bo_target']
    ).reset_index(drop=True)


def filter_hf_training_rows(hf_data, split_col='split',
                            exclude_splits='blind'):
    """Remove held-out HF rows from GP training."""
    if hf_data is None:
        return None
    if not split_col or split_col not in hf_data:
        return hf_data

    blocked = {
        item.strip()
        for item in str(exclude_splits).split(',')
        if item.strip()
    }
    if not blocked:
        return hf_data

    keep = ~hf_data[split_col].astype(str).isin(blocked)
    return hf_data.loc[keep].reset_index(drop=True)


def add_pgnn_rank(candidates):
    """Sort candidates by PGNN prediction and add pgnn_rank."""
    ranked = candidates.sort_values('k_pred', ascending=False).reset_index(
        drop=True
    )
    ranked.insert(0, 'pgnn_rank', np.arange(1, len(ranked) + 1))
    return ranked


def select_pgnn_shortlist(ranked_candidates, pgnn_top_n=10):
    """Take the PGNN top-N shortlist before BO screening."""
    if pgnn_top_n <= 0:
        return ranked_candidates.copy()
    return ranked_candidates.head(pgnn_top_n).reset_index(drop=True)


def build_acquisition_pool(ranked_candidates, shortlist, pool_mode):
    """Choose which rows BO should rank."""
    if pool_mode == 'top-pgnn':
        return shortlist.copy()
    if pool_mode == 'all':
        return ranked_candidates.copy()
    msg = 'Unknown candidate pool mode: {}'.format(pool_mode)
    raise ValueError(msg)


def resolve_gp_training_data(
    ranked_candidates,
    shortlist,
    hf_data,
    gp_train_source='auto',
):
    """Resolve GP training features and target values."""
    source = gp_train_source
    if source == 'auto':
        source = 'hf' if hf_data is not None and not hf_data.empty else (
            'pgnn_top'
        )

    if source == 'hf':
        if hf_data is None or hf_data.empty:
            msg = 'gp_train_source=hf requires non-empty --hf-data.'
            raise ValueError(msg)
        train_frame = hf_data.copy()
        y_train = train_frame['bo_target'].to_numpy(dtype=float)
    elif source == 'pgnn_top':
        train_frame = shortlist.copy()
        y_train = train_frame['k_pred'].to_numpy(dtype=float)
    elif source == 'pgnn_all':
        train_frame = ranked_candidates.copy()
        y_train = train_frame['k_pred'].to_numpy(dtype=float)
    else:
        msg = 'Unknown gp_train_source: {}'.format(gp_train_source)
        raise ValueError(msg)

    if len(train_frame) < 2:
        msg = 'Bayesian optimization needs at least two GP training rows.'
        raise ValueError(msg)

    return train_frame, y_train, source


def fit_gp(train_frame, y_train, args):
    """Fit a Gaussian Process surrogate on transformed features."""
    x_train = make_bo_feature_matrix(train_frame)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            length_scale=np.ones(x_scaled.shape[1]),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=args.alpha,
            noise_level_bounds=(1e-9, 1e0),
        )
    )
    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=args.alpha,
        normalize_y=True,
        n_restarts_optimizer=args.n_restarts_optimizer,
        random_state=args.seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        gp_model.fit(x_scaled, y_train)
    return gp_model, scaler


def score_acquisition(pool, gp_model, scaler, y_best, args):
    """Calculate GP posterior, EI, and final BO score."""
    x_pool = make_bo_feature_matrix(pool)
    mean, std = gp_model.predict(scaler.transform(x_pool), return_std=True)

    scored = pool.copy().reset_index(drop=True)
    scored['gp_mean'] = mean
    scored['gp_std'] = std
    scored['expected_improvement'] = expected_improvement(
        mean,
        std,
        y_best,
        xi=args.xi,
    )
    scored['gp_ucb'] = mean + args.ucb_kappa * std
    scored['normalized_ei'] = normalize_series(
        scored['expected_improvement']
    )
    scored['normalized_gp_mean'] = normalize_series(scored['gp_mean'])
    scored['normalized_gp_ucb'] = normalize_series(scored['gp_ucb'])
    scored['normalized_E'] = normalize_series(scored['E'])
    scored['normalized_k_pred'] = normalize_series(scored['k_pred'])
    scored['grading_error_penalty'] = args.error_weight * scored[
        'normalized_E'
    ]
    scored['pgnn_score_bonus'] = args.pgnn_score_weight * scored[
        'normalized_k_pred'
    ]
    scored['bo_base_score'] = resolve_acquisition_score(scored, args)
    scored['bo_score'] = (
        scored['bo_base_score']
        - scored['grading_error_penalty']
        + scored['pgnn_score_bonus']
    )
    scored['bo_y_best'] = y_best
    return scored


def filter_measured_candidates(pool, hf_data, key_precision=6):
    """Remove candidates that already match measured HF formulations."""
    if hf_data is None or hf_data.empty:
        return pool.reset_index(drop=True)

    measured_keys = set(make_feature_keys(hf_data, precision=key_precision))
    candidate_keys = make_feature_keys(pool, precision=key_precision)
    keep = [key not in measured_keys for key in candidate_keys]
    return pool.loc[keep].reset_index(drop=True)


def select_bo_top(scored, bo_top_n=5, max_per_triplet=2):
    """Select final BO suggestions with a per-triplet diversity cap."""
    if bo_top_n <= 0:
        msg = 'bo_top_n must be greater than zero.'
        raise ValueError(msg)
    if max_per_triplet <= 0:
        msg = 'max_per_triplet must be greater than zero.'
        raise ValueError(msg)

    ranked = scored.sort_values(
        ['bo_score', 'expected_improvement', 'k_pred'],
        ascending=[False, False, False],
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
        if len(selected) >= bo_top_n:
            break

    if not selected:
        return ranked.head(0).copy()

    output = pd.DataFrame(selected).reset_index(drop=True)
    output.insert(0, 'bo_rank', np.arange(1, len(output) + 1))
    return order_bo_columns(output)


def resolve_acquisition_score(scored, args):
    """Resolve the acquisition score used for final ranking."""
    if args.acquisition == 'ei':
        return scored['normalized_ei']
    if args.acquisition == 'ucb':
        return scored['normalized_gp_ucb']
    if args.acquisition == 'hybrid':
        return 0.5 * scored['normalized_ei'] + 0.5 * scored[
            'normalized_gp_ucb'
        ]

    msg = 'Unknown acquisition: {}'.format(args.acquisition)
    raise ValueError(msg)


def expected_improvement(mean, std, best_value, xi=0.01):
    """Calculate expected improvement for maximization."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    improvement = mean - best_value - xi
    ei = np.zeros_like(mean)

    mask = std > 1e-12
    if mask.any():
        z = improvement[mask] / std[mask]
        ei[mask] = (
            improvement[mask] * normal_cdf(z)
            + std[mask] * normal_pdf(z)
        )
    if (~mask).any():
        ei[~mask] = np.maximum(improvement[~mask], 0.0)

    return np.maximum(ei, 0.0)


def normal_pdf(z):
    """Standard normal probability density function."""
    return np.exp(-0.5 * z**2) / SQRT_TWO_PI


def normal_cdf(z):
    """Standard normal cumulative distribution function."""
    erf_vec = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(z / SQRT_TWO))


def normalize_series(values):
    """Normalize values to [0, 1] while handling constant columns."""
    values = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return np.zeros_like(values)
    if max_value <= min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def order_bo_columns(output):
    """Put the most important BO columns first."""
    first_cols = [
        'bo_rank',
        'bo_score',
        'expected_improvement',
        'bo_base_score',
        'gp_mean',
        'gp_std',
        'gp_ucb',
        'bo_y_best',
        'gp_train_source',
        'gp_train_n',
        'candidate_pool',
        'normalized_ei',
        'normalized_gp_ucb',
        'pgnn_rank',
        'k_pred',
        'Delta_k_pred',
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
        'normalized_E',
        'normalized_k_pred',
        'grading_error_penalty',
        'pgnn_score_bonus',
    ]
    ordered = [col for col in first_cols if col in output.columns]
    remaining = [col for col in output.columns if col not in ordered]
    return output[ordered + remaining]


def main(argv=None):
    """Run PGNN top-N plus BO screening."""
    parser = build_parser()
    args = parser.parse_args(argv)

    candidates = load_candidates(args.candidates)
    model = maybe_load_model(args.model, no_model=args.no_model)
    candidates = ensure_pgnn_predictions(
        candidates,
        model=model,
        refresh=args.refresh_pgnn_predictions,
    )
    ranked_candidates = add_pgnn_rank(candidates)
    shortlist = select_pgnn_shortlist(ranked_candidates, args.pgnn_top_n)

    hf_all = load_hf_data(args.hf_data, label_col=args.hf_label_col)
    hf_train = filter_hf_training_rows(
        hf_all,
        split_col=args.hf_split_col,
        exclude_splits=args.exclude_hf_splits,
    )

    pool = build_acquisition_pool(
        ranked_candidates,
        shortlist,
        args.candidate_pool,
    )
    if not args.no_exclude_measured:
        pool = filter_measured_candidates(
            pool,
            hf_all,
            key_precision=args.key_precision,
        )
    if pool.empty:
        msg = 'No candidates remain after filtering measured formulations.'
        raise ValueError(msg)

    train_frame, y_train, gp_source = resolve_gp_training_data(
        ranked_candidates,
        shortlist,
        hf_train,
        gp_train_source=args.gp_train_source,
    )
    if gp_source.startswith('pgnn'):
        print(
            'BO is using PGNN predictions as pseudo-observations; '
            'add --hf-data for true experiment-feedback BO.'
        )

    gp_model, scaler = fit_gp(train_frame, y_train, args)
    scored = score_acquisition(
        pool,
        gp_model,
        scaler,
        y_best=float(np.max(y_train)),
        args=args,
    )
    scored['gp_train_source'] = gp_source
    scored['gp_train_n'] = len(train_frame)
    scored['candidate_pool'] = args.candidate_pool
    suggestions = select_bo_top(
        scored,
        bo_top_n=args.bo_top_n,
        max_per_triplet=args.max_per_triplet,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suggestions.to_csv(args.output, index=False, encoding='utf-8-sig')

    preview_cols = [
        'bo_rank',
        'pgnn_rank',
        'triplet_id',
        'D_s',
        'D_m',
        'D_l',
        'phi_s',
        'phi_m',
        'E',
        'k_pred',
        'expected_improvement',
        'bo_score',
    ]
    preview_cols = [col for col in preview_cols if col in suggestions]
    print('Saved {} BO suggestions to {}'.format(len(suggestions),
                                                 args.output.resolve()))
    print(suggestions[preview_cols].head(args.bo_top_n).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
