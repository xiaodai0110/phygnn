"""Convert PGNN candidate formulations into weighable lab recipes.

The PGNN works with volume fractions and particle-size descriptors. Real
experiments need gram-level masses. This script converts candidate rows into
per-repeat recipes with Al2O3 small/medium/large powder masses and matrix
system masses.

Expected input columns:
    D_s,D_m,D_l,phi_s,phi_m,E

Optional columns:
    phi_l,V_f,k_pred,triplet_id,small_id,medium_id,large_id
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phygnn.bn_al2o3.constants import OPTIONAL_INPUT_COLUMNS, RAW_FEATURE_NAMES
from phygnn.bn_al2o3.paths import outputs_path


def build_parser():
    """Build command-line arguments for recipe generation."""
    parser = argparse.ArgumentParser(
        description='Convert BN/Al2O3 candidate rows into lab recipes.'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=outputs_path('bn_al2o3_predictions.csv'),
        help='Candidate or prediction CSV.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=outputs_path('experiment_recipes.csv'),
        help='Output recipe CSV.',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=0,
        help='Use only the first/top N formulations. Use <=0 for all.',
    )
    parser.add_argument(
        '--sort-by',
        default=None,
        help='Column used before top-N selection. Defaults to k_pred or E.',
    )
    parser.add_argument(
        '--ascending',
        action='store_true',
        help='Sort ascending instead of the default descending order.',
    )
    parser.add_argument('--repeat-count', type=int, default=3)
    parser.add_argument('--batch-mass-g', type=float, default=20.0)
    parser.add_argument('--vf', type=float, default=0.30)
    parser.add_argument('--al2o3-density', type=float, default=3.95)
    parser.add_argument('--matrix-density', type=float, default=1.15)
    parser.add_argument(
        '--bn-mass-fraction-in-matrix',
        type=float,
        default=0.0,
    )
    parser.add_argument('--bn-mass-g', type=float, default=None)
    parser.add_argument('--hardener-per-resin', type=float, default=0.0)
    parser.add_argument('--id-prefix', default='EXP')
    return parser


def load_candidate_table(path, default_vf=0.30):
    """Load and normalize candidate rows."""
    frame = pd.read_csv(path)
    return normalize_candidate_table(frame, default_vf=default_vf)


def normalize_candidate_table(frame, default_vf=0.30):
    """Validate candidate features and complete phi_l/V_f columns."""
    data = frame.copy()
    missing = [col for col in RAW_FEATURE_NAMES if col not in data.columns]
    if missing:
        msg = 'Candidate table is missing columns: {}'.format(
            ', '.join(missing)
        )
        raise ValueError(msg)

    if 'phi_l' not in data.columns:
        data['phi_l'] = 1.0 - data['phi_s'] - data['phi_m']
    if 'V_f' not in data.columns:
        data['V_f'] = default_vf

    numeric_cols = RAW_FEATURE_NAMES + ['phi_l', 'V_f']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    invalid = (
        data[numeric_cols].isna().any(axis=1)
        | (data[['D_s', 'D_m', 'D_l']] <= 0).any(axis=1)
        | (data[['phi_s', 'phi_m', 'phi_l']] < 0).any(axis=1)
        | (data['V_f'] <= 0)
        | (data['V_f'] >= 1)
    )
    if invalid.any():
        bad_rows = ', '.join(map(str, data.index[invalid].tolist()[:10]))
        msg = 'Invalid candidate rows at indices: {}'.format(bad_rows)
        raise ValueError(msg)

    # Excel 保存后配比可能存在很小误差，这里统一归一化。
    phi_sum = data[['phi_s', 'phi_m', 'phi_l']].sum(axis=1)
    data[['phi_s', 'phi_m', 'phi_l']] = data[
        ['phi_s', 'phi_m', 'phi_l']
    ].div(phi_sum, axis=0)
    return data.reset_index(drop=True)


def create_experiment_recipes(
    candidates,
    *,
    repeat_count=3,
    batch_mass_g=20.0,
    al2o3_density=3.95,
    matrix_density=1.15,
    bn_mass_fraction_in_matrix=0.0,
    bn_mass_g=None,
    hardener_per_resin=0.0,
    id_prefix='EXP',
):
    """Create per-repeat weighable recipes from candidate rows."""
    _validate_recipe_settings(
        repeat_count,
        batch_mass_g,
        al2o3_density,
        matrix_density,
        bn_mass_fraction_in_matrix,
        bn_mass_g,
        hardener_per_resin,
    )
    rows = []

    for form_idx, row in candidates.reset_index(drop=True).iterrows():
        formulation_id = 'FORM_{:05d}'.format(form_idx + 1)
        mass_row = _calculate_masses(
            row,
            batch_mass_g=batch_mass_g,
            al2o3_density=al2o3_density,
            matrix_density=matrix_density,
            bn_mass_fraction_in_matrix=bn_mass_fraction_in_matrix,
            bn_mass_g=bn_mass_g,
            hardener_per_resin=hardener_per_resin,
        )

        for repeat_idx in range(1, repeat_count + 1):
            recipe = {
                'experiment_id': '{}_{:05d}_R{:02d}'.format(
                    id_prefix,
                    form_idx + 1,
                    repeat_idx,
                ),
                'formulation_id': formulation_id,
                'repeat_id': repeat_idx,
                'source_row': int(form_idx),
            }
            recipe.update(_copy_candidate_metadata(row))
            recipe.update(mass_row)
            recipe['mixing_notes'] = 'weigh powders before adding matrix'
            rows.append(recipe)

    return _order_recipe_columns(pd.DataFrame(rows))


def select_candidate_rows(candidates, top_n=0, sort_by=None, ascending=False):
    """Sort and select candidate rows before recipe conversion."""
    data = candidates.copy()
    sort_col, sort_ascending = _resolve_sort(data, sort_by, ascending)
    if sort_col is not None:
        data = data.sort_values(sort_col, ascending=sort_ascending)

    if top_n is not None and top_n > 0:
        data = data.head(top_n)
    return data.reset_index(drop=True)


def main(argv=None):
    """Run the recipe-generation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    candidates = load_candidate_table(args.input, default_vf=args.vf)
    candidates = select_candidate_rows(
        candidates,
        top_n=args.top_n,
        sort_by=args.sort_by,
        ascending=args.ascending,
    )
    recipes = create_experiment_recipes(
        candidates,
        repeat_count=args.repeat_count,
        batch_mass_g=args.batch_mass_g,
        al2o3_density=args.al2o3_density,
        matrix_density=args.matrix_density,
        bn_mass_fraction_in_matrix=args.bn_mass_fraction_in_matrix,
        bn_mass_g=args.bn_mass_g,
        hardener_per_resin=args.hardener_per_resin,
        id_prefix=args.id_prefix,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    recipes.to_csv(args.output, index=False, encoding='utf-8-sig')

    preview_cols = [
        'experiment_id',
        'formulation_id',
        'repeat_id',
        'al2o3_s_mass_g',
        'al2o3_m_mass_g',
        'al2o3_l_mass_g',
        'resin_mass_g',
        'hardener_mass_g',
        'k_pred',
    ]
    preview_cols = [col for col in preview_cols if col in recipes.columns]
    print('Saved {} recipe rows to {}'.format(len(recipes),
                                              args.output.resolve()))
    print(recipes[preview_cols].head(10).to_string(index=False))
    return 0


def _calculate_masses(
    row,
    *,
    batch_mass_g,
    al2o3_density,
    matrix_density,
    bn_mass_fraction_in_matrix,
    bn_mass_g,
    hardener_per_resin,
):
    V_f = float(row['V_f'])
    total_volume_cm3 = batch_mass_g / (
        al2o3_density * V_f + matrix_density * (1.0 - V_f)
    )
    al2o3_total_mass = al2o3_density * V_f * total_volume_cm3
    matrix_mass = matrix_density * (1.0 - V_f) * total_volume_cm3

    if bn_mass_g is None:
        bn_mass = matrix_mass * bn_mass_fraction_in_matrix
    else:
        bn_mass = bn_mass_g
    polymer_system_mass = matrix_mass - bn_mass
    if polymer_system_mass < 0:
        msg = 'BN mass is larger than calculated matrix mass.'
        raise ValueError(msg)

    resin_mass = polymer_system_mass / (1.0 + hardener_per_resin)
    hardener_mass = polymer_system_mass - resin_mass

    return {
        'target_batch_mass_g': batch_mass_g,
        'total_volume_cm3': total_volume_cm3,
        'V_f': V_f,
        'al2o3_density_g_cm3': al2o3_density,
        'matrix_density_g_cm3': matrix_density,
        'al2o3_total_mass_g': al2o3_total_mass,
        'al2o3_s_mass_g': al2o3_total_mass * row['phi_s'],
        'al2o3_m_mass_g': al2o3_total_mass * row['phi_m'],
        'al2o3_l_mass_g': al2o3_total_mass * row['phi_l'],
        'matrix_mass_g': matrix_mass,
        'bn_mass_g': bn_mass,
        'resin_mass_g': resin_mass,
        'hardener_mass_g': hardener_mass,
    }


def _copy_candidate_metadata(row):
    cols = RAW_FEATURE_NAMES + ['phi_l', 'V_f'] + OPTIONAL_INPUT_COLUMNS
    copied = {}
    for col in cols:
        if col in row.index:
            copied[col] = row[col]
    return copied


def _validate_recipe_settings(
    repeat_count,
    batch_mass_g,
    al2o3_density,
    matrix_density,
    bn_mass_fraction_in_matrix,
    bn_mass_g,
    hardener_per_resin,
):
    if repeat_count < 1:
        msg = 'repeat_count must be at least 1.'
        raise ValueError(msg)
    if batch_mass_g <= 0 or al2o3_density <= 0 or matrix_density <= 0:
        msg = 'batch mass and densities must be positive.'
        raise ValueError(msg)
    if not 0 <= bn_mass_fraction_in_matrix < 1:
        msg = 'bn_mass_fraction_in_matrix must be in [0, 1).'
        raise ValueError(msg)
    if bn_mass_g is not None and bn_mass_g < 0:
        msg = 'bn_mass_g must be non-negative.'
        raise ValueError(msg)
    if hardener_per_resin < 0:
        msg = 'hardener_per_resin must be non-negative.'
        raise ValueError(msg)


def _resolve_sort(data, sort_by, ascending):
    if sort_by is not None:
        if sort_by not in data.columns:
            msg = 'Sort column not found: {}'.format(sort_by)
            raise ValueError(msg)
        return sort_by, ascending
    if 'k_pred' in data.columns:
        return 'k_pred', ascending
    if 'E' in data.columns:
        return 'E', True
    return None, ascending


def _order_recipe_columns(recipes):
    first_cols = [
        'experiment_id',
        'formulation_id',
        'repeat_id',
        'source_row',
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
        'k',
        'k_LF',
        'target_batch_mass_g',
        'total_volume_cm3',
        'al2o3_total_mass_g',
        'al2o3_s_mass_g',
        'al2o3_m_mass_g',
        'al2o3_l_mass_g',
        'matrix_mass_g',
        'bn_mass_g',
        'resin_mass_g',
        'hardener_mass_g',
        'al2o3_density_g_cm3',
        'matrix_density_g_cm3',
        'mixing_notes',
    ]
    ordered = [col for col in first_cols if col in recipes.columns]
    remaining = [col for col in recipes.columns if col not in ordered]
    return recipes[ordered + remaining]


if __name__ == '__main__':
    raise SystemExit(main())
