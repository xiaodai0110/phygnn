"""Prepare high-fidelity BN/Al2O3 measurement data for PGNN training.

This script turns raw lab measurements into a clean high-fidelity table:

1. Merge measured k values with experiment recipes when needed.
2. Validate PGNN feature columns.
3. Mark or remove repeat-level outliers.
4. Average repeated measurements by formulation.
5. Optionally assign dev/blind split labels.

Training output columns include:
    D_s,D_m,D_l,phi_s,phi_m,E,k_HF,k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from phygnn.bn_al2o3.constants import (
    MERGE_KEY_CANDIDATES,
    METADATA_COLUMNS,
    OBSERVED_K_CANDIDATES,
    RAW_FEATURE_NAMES,
)
from phygnn.bn_al2o3.features import infer_column
from phygnn.bn_al2o3.paths import data_path


def build_parser():
    """Build command-line arguments for HF data preparation."""
    parser = argparse.ArgumentParser(
        description='Prepare measured HF BN/Al2O3 data for PGNN training.'
    )
    parser.add_argument('--raw-measurements', type=Path, default=None)
    parser.add_argument('--recipes', type=Path, default=None)
    parser.add_argument('--merge-key', default=None)
    parser.add_argument('--label-col', default=None)
    parser.add_argument(
        '--output',
        type=Path,
        default=data_path('hf_bn_al2o3_data.csv'),
    )
    parser.add_argument('--repeat-output', type=Path, default=None)
    parser.add_argument('--write-template', type=Path, default=None)
    parser.add_argument('--outlier-z', type=float, default=3.0)
    parser.add_argument('--drop-outliers', action='store_true')
    parser.add_argument('--blind-count', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def write_template(path):
    """Write a raw HF measurement template CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame(
        [
            {
                'experiment_id': 'EXP_00001_R01',
                'formulation_id': 'FORM_00001',
                'repeat_id': 1,
                'D_s': 1.2,
                'D_m': 8.0,
                'D_l': 25.0,
                'phi_s': 0.20,
                'phi_m': 0.30,
                'phi_l': 0.50,
                'E': 0.00015,
                'V_f': 0.30,
                'k_HF': 3.20,
                'measurement_notes': 'replace with measured data',
            }
        ]
    )
    template.to_csv(path, index=False, encoding='utf-8-sig')


def load_hf_measurements(raw_path, recipes_path=None, merge_key=None):
    """Load raw measurements and optionally merge recipe metadata."""
    if raw_path is None:
        msg = 'Provide --raw-measurements or use --write-template.'
        raise FileNotFoundError(msg)
    raw = pd.read_csv(raw_path)
    if recipes_path is None:
        return raw

    recipes = pd.read_csv(recipes_path)
    key = resolve_merge_key(raw, recipes, merge_key)
    recipe_cols = [key]
    recipe_cols += [
        col for col in METADATA_COLUMNS if col in recipes and col not in raw
    ]
    merged = raw.merge(recipes[recipe_cols], on=key, how='left')
    return merged


def resolve_merge_key(raw, recipes, merge_key=None):
    """Resolve the ID column used to merge raw measurements and recipes."""
    if merge_key is not None:
        if merge_key not in raw or merge_key not in recipes:
            msg = 'merge_key must exist in both raw and recipe tables.'
            raise ValueError(msg)
        return merge_key

    for col in MERGE_KEY_CANDIDATES:
        if col in raw and col in recipes:
            return col

    msg = 'Could not find a shared merge key between raw and recipe tables.'
    raise ValueError(msg)


def normalize_hf_rows(frame, label_col=None):
    """Validate repeat-level HF rows and standardize the k_HF column."""
    data = frame.copy()
    label_col = label_col or infer_column(
        data,
        OBSERVED_K_CANDIDATES,
        'measured conductivity',
    )
    data['k_HF'] = pd.to_numeric(data[label_col], errors='coerce')

    if 'phi_l' not in data.columns and {'phi_s', 'phi_m'} <= set(data.columns):
        data['phi_l'] = 1.0 - data['phi_s'] - data['phi_m']

    missing = [col for col in RAW_FEATURE_NAMES if col not in data.columns]
    if missing:
        msg = 'HF rows are missing feature columns: {}'.format(
            ', '.join(missing)
        )
        raise ValueError(msg)

    numeric_cols = RAW_FEATURE_NAMES + ['k_HF']
    if 'phi_l' in data.columns:
        numeric_cols.append('phi_l')
    if 'V_f' in data.columns:
        numeric_cols.append('V_f')

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    invalid = (
        data[numeric_cols].isna().any(axis=1)
        | (data[['D_s', 'D_m', 'D_l']] <= 0).any(axis=1)
        | (data[['phi_s', 'phi_m']] < 0).any(axis=1)
        | (data['E'] < 0)
        | (data['k_HF'] <= 0)
    )
    if 'phi_l' in data.columns:
        invalid = invalid | (data['phi_l'] < 0)

    if invalid.any():
        bad_rows = ', '.join(map(str, data.index[invalid].tolist()[:10]))
        msg = 'Invalid HF rows at indices: {}'.format(bad_rows)
        raise ValueError(msg)

    data['k'] = data['k_HF']
    return data.reset_index(drop=True)


def mark_outliers(frame, outlier_z=3.0):
    """Add repeat-level z-score and outlier flag columns."""
    if outlier_z <= 0:
        msg = 'outlier_z must be positive.'
        raise ValueError(msg)

    data = frame.copy()
    group_cols = _repeat_group_columns(data)
    grouped = data.groupby(group_cols)['k_HF']
    mean = grouped.transform('mean')
    std = grouped.transform('std').fillna(0.0)
    count = grouped.transform('size')

    z_score = np.zeros(len(data), dtype=float)
    mask = std > 0
    z_score[mask] = np.abs((data.loc[mask, 'k_HF'] - mean[mask]) / std[mask])
    data['hf_repeat_zscore'] = z_score
    data['hf_outlier'] = (count >= 3) & (data['hf_repeat_zscore'] > outlier_z)
    return data


def aggregate_hf_data(frame, blind_count=0, seed=42):
    """Average repeat-level HF measurements into formulation-level rows."""
    group_cols = RAW_FEATURE_NAMES
    agg_spec = {
        'k_HF': ('k_HF', 'mean'),
        'k_HF_std': ('k_HF', 'std'),
        'hf_repeat_count': ('k_HF', 'size'),
    }
    for col in METADATA_COLUMNS:
        if col in frame.columns and col not in group_cols:
            agg_spec[col] = (col, 'first')

    summary = frame.groupby(group_cols, as_index=False).agg(**agg_spec)
    summary['k_HF_std'] = summary['k_HF_std'].fillna(0.0)
    summary['k'] = summary['k_HF']
    summary = assign_splits(summary, blind_count=blind_count, seed=seed)
    return _order_summary_columns(summary)


def assign_splits(summary, blind_count=0, seed=42):
    """Assign dev/blind split labels at formulation level."""
    output = summary.copy()
    if 'split' not in output.columns:
        output['split'] = 'dev'

    if blind_count is None or blind_count <= 0:
        return output

    rng = np.random.default_rng(seed)
    n_blind = min(blind_count, len(output))
    blind_idx = rng.choice(output.index.to_numpy(), size=n_blind,
                           replace=False)
    output['split'] = 'dev'
    output.loc[blind_idx, 'split'] = 'blind'
    return output


def main(argv=None):
    """Run the HF preparation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_template is not None:
        write_template(args.write_template)
        print('Template written to {}'.format(args.write_template.resolve()))
        return 0

    raw = load_hf_measurements(
        args.raw_measurements,
        recipes_path=args.recipes,
        merge_key=args.merge_key,
    )
    repeats = normalize_hf_rows(raw, label_col=args.label_col)
    repeats = mark_outliers(repeats, outlier_z=args.outlier_z)
    training_repeats = repeats.loc[~repeats['hf_outlier']].copy()
    if not args.drop_outliers:
        training_repeats = repeats.copy()

    summary = aggregate_hf_data(
        training_repeats,
        blind_count=args.blind_count,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding='utf-8-sig')
    if args.repeat_output is not None:
        args.repeat_output.parent.mkdir(parents=True, exist_ok=True)
        repeats.to_csv(args.repeat_output, index=False, encoding='utf-8-sig')

    preview_cols = [
        'formulation_id',
        'D_s',
        'D_m',
        'D_l',
        'phi_s',
        'phi_m',
        'E',
        'k_HF',
        'k_HF_std',
        'hf_repeat_count',
        'split',
    ]
    preview_cols = [col for col in preview_cols if col in summary.columns]
    print('Saved {} HF formulation rows to {}'.format(len(summary),
                                                      args.output.resolve()))
    print(summary[preview_cols].head(10).to_string(index=False))
    return 0


def _repeat_group_columns(data):
    if 'formulation_id' in data.columns:
        return ['formulation_id']
    return RAW_FEATURE_NAMES


def _order_summary_columns(summary):
    first_cols = [
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
        'k_HF',
        'k',
        'k_HF_std',
        'hf_repeat_count',
        'split',
        'rank',
        'k_pred',
    ]
    ordered = [col for col in first_cols if col in summary.columns]
    remaining = [col for col in summary.columns if col not in ordered]
    return summary[ordered + remaining]


if __name__ == '__main__':
    raise SystemExit(main())
