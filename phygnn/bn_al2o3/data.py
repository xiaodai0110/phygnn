# -*- coding: utf-8 -*-
"""Data pipeline for the BN/Al2O3 workflow.

This module owns the pandas/CSV/DataFrame side of the workflow: powder-table
loading, Dinger-Funk triplet selection, and Lewis-Nielsen low-fidelity data
generation. Pure equations live in :mod:`phygnn.bn_al2o3.physics`.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from phygnn.bn_al2o3.constants import LABEL_NAMES, RAW_FEATURE_NAMES
from phygnn.bn_al2o3.physics import (
    _clip_phi_max,
    _integrated_squared_error,
    _lognormal_cdf_from_quantiles,
    clip_volume_fraction,
    dinger_funk_target,
    estimate_phi_max_from_error,
    lewis_nielsen_conductivity,
    make_ratio_grid,
    sample_volume_fraction,
)

# 输出列里保留了后续 PGNN 训练直接需要的核心字段：
# D_s/D_m/D_l, phi_s/phi_m/phi_l, E。
RESULT_COLUMNS = [
    'rank',
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
    'df_exponent',
    'd_min',
    'd_max',
    'size_ratio_sm',
    'size_ratio_ml',
]

# DF 候选表进入 LF 生成前必须包含的列。
REQUIRED_CANDIDATE_COLUMNS = [
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
]


def demo_powder_table():
    """Return a small example Al2O3 powder library."""
    # 没有真实粉体库 CSV 时，用这组示例值检查脚本是否能跑通。
    # 真实筛选时应该换成供应商或实测的 Al2O3 粉体 PSD 数据。
    return pd.DataFrame(
        [
            {'powder_id': 'Al2O3_0p5', 'D10': 0.28, 'D50': 0.50, 'D90': 0.90},
            {'powder_id': 'Al2O3_1p2', 'D10': 0.70, 'D50': 1.20, 'D90': 2.10},
            {'powder_id': 'Al2O3_3p0', 'D10': 1.80, 'D50': 3.00, 'D90': 5.20},
            {'powder_id': 'Al2O3_8p0', 'D10': 4.80, 'D50': 8.00, 'D90': 14.0},
            {'powder_id': 'Al2O3_25', 'D10': 15.0, 'D50': 25.0, 'D90': 42.0},
            {'powder_id': 'Al2O3_70', 'D10': 42.0, 'D50': 70.0, 'D90': 115.0},
        ]
    )


def normalize_powder_table(powders):
    """Normalize powder library columns and validate particle sizes."""
    raw = powders.copy()

    # 支持大小写不同或少量不同的列名，方便接入不同来源的 CSV。
    d10_col = _resolve_column(raw, ('D10', 'd10'))
    d50_col = _resolve_column(raw, ('D50', 'd50', 'median_diameter'))
    d90_col = _resolve_column(raw, ('D90', 'd90'))
    id_col = _find_optional_column(raw, ('powder_id', 'id', 'name'))

    # 如果用户没有提供粉体 ID，就自动生成一个稳定编号。
    if id_col is None:
        ids = ['powder_{:03d}'.format(i + 1) for i in range(len(raw))]
    else:
        ids = raw[id_col].astype(str).to_numpy()

    # 把 D10/D50/D90 转成数值列，后面统一用这三个分位点建 PSD。
    table = pd.DataFrame(
        {
            'powder_id': ids,
            'D10': pd.to_numeric(raw[d10_col], errors='coerce'),
            'D50': pd.to_numeric(raw[d50_col], errors='coerce'),
            'D90': pd.to_numeric(raw[d90_col], errors='coerce'),
        }
    )

    # 合法粉体必须满足 0 < D10 < D50 < D90。
    # 不满足这个关系时，对数正态近似和 DF 误差都会失去意义。
    invalid = (
        table[['D10', 'D50', 'D90']].isna().any(axis=1)
        | (table[['D10', 'D50', 'D90']] <= 0).any(axis=1)
        | (table['D10'] >= table['D50'])
        | (table['D50'] >= table['D90'])
    )
    if invalid.any():
        bad_ids = ', '.join(table.loc[invalid, 'powder_id'].astype(str))
        msg = 'Invalid powder rows. Need 0 < D10 < D50 < D90: {}'.format(
            bad_ids
        )
        raise ValueError(msg)

    return table.sort_values('D50').reset_index(drop=True)


def load_powder_table(path):
    """Load and validate a powder library CSV."""
    return normalize_powder_table(pd.read_csv(path))


def write_template(path):
    """Write a CSV template for the powder library."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 模板沿用 demo 数据，用户只需要替换为真实 D10/D50/D90 即可。
    template = demo_powder_table()
    template['notes'] = 'replace example values with measured/vendor PSD data'
    template.to_csv(path, index=False, encoding='utf-8-sig')


def select_df_triplets(
    powders,
    *,
    df_exponent=0.37,
    ratio_step=0.05,
    min_fraction=0.05,
    min_size_ratio=1.2,
    grid_points=240,
    top_n=30,
    d_min=None,
    d_max=None,
    max_error=None,
    keep_all_ratios=False,
):
    """Rank particle-size triplets by Dinger-Funk PSD matching error."""
    # 先统一粉体表，再生成所有可能的三峰配比候选。
    table = normalize_powder_table(powders)
    ratios = make_ratio_grid(ratio_step, min_fraction)
    rows = []

    # 粉体表已经按 D50 从小到大排序。
    # combinations 取出的三元组自然对应 small/medium/large。
    for small_i, medium_i, large_i in combinations(range(len(table)), 3):
        small = table.iloc[small_i]
        medium = table.iloc[medium_i]
        large = table.iloc[large_i]

        # 相邻粒径太接近时，三峰级配的物理意义不明显，先过滤掉。
        size_ratio_sm = medium['D50'] / small['D50']
        size_ratio_ml = large['D50'] / medium['D50']
        if (
            size_ratio_sm < min_size_ratio
            or size_ratio_ml < min_size_ratio
        ):
            continue

        # 默认把该 triplet 的最小 D10 和最大 D90 作为 DF 目标范围。
        # 也可以通过 --d-min/--d-max 固定整个实验的统一范围。
        d_min_use = float(d_min) if d_min is not None else float(small['D10'])
        d_max_use = float(d_max) if d_max is not None else float(large['D90'])
        if d_max_use <= d_min_use:
            continue

        # 在对数粒径坐标上取点，细粉和粗粉区域都有足够分辨率。
        diameters = np.geomspace(d_min_use, d_max_use, grid_points)
        target = dinger_funk_target(
            diameters,
            d_min_use,
            d_max_use,
            df_exponent,
        )

        # 用 D10/D50/D90 把每一种粉体近似成对数正态累计 PSD。
        psds = np.vstack(
            [
                _lognormal_cdf_from_quantiles(diameters, small),
                _lognormal_cdf_from_quantiles(diameters, medium),
                _lognormal_cdf_from_quantiles(diameters, large),
            ]
        )

        # ratios @ psds 会一次性得到所有候选配比的混合累计 PSD。
        # E 是混合 PSD 与 DF 目标 PSD 的积分平方误差。
        errors = _integrated_squared_error(ratios @ psds, target, diameters)

        # 默认每个 triplet 只保留误差最小的配比。
        # 若打开 --keep-all-ratios，则输出所有配比候选，便于人工复查。
        if keep_all_ratios:
            ratio_indices = range(len(ratios))
        else:
            ratio_indices = [int(np.argmin(errors))]

        for ratio_idx in ratio_indices:
            error = float(errors[ratio_idx])
            if max_error is not None and error > max_error:
                continue

            phi_s, phi_m, phi_l = ratios[ratio_idx]
            rows.append(
                {
                    'triplet_id': '{}|{}|{}'.format(
                        small['powder_id'],
                        medium['powder_id'],
                        large['powder_id'],
                    ),
                    'small_id': small['powder_id'],
                    'medium_id': medium['powder_id'],
                    'large_id': large['powder_id'],
                    'D_s': float(small['D50']),
                    'D_m': float(medium['D50']),
                    'D_l': float(large['D50']),
                    'phi_s': float(phi_s),
                    'phi_m': float(phi_m),
                    'phi_l': float(phi_l),
                    'E': error,
                    'df_exponent': float(df_exponent),
                    'd_min': d_min_use,
                    'd_max': d_max_use,
                    'size_ratio_sm': float(size_ratio_sm),
                    'size_ratio_ml': float(size_ratio_ml),
                }
            )

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    # 按 E 从小到大排序；E 越小，表示越接近 DF 理想级配。
    results = pd.DataFrame(rows).sort_values(['E', 'triplet_id'])
    results = results.reset_index(drop=True)
    results.insert(0, 'rank', np.arange(1, len(results) + 1))
    if top_n is not None and top_n > 0:
        results = results.head(top_n)

    return results[RESULT_COLUMNS]


def load_df_candidates(path):
    """Load DF candidate rows created by the particle-size selector."""
    candidates = pd.read_csv(path)
    return normalize_df_candidates(candidates)


def normalize_df_candidates(candidates):
    """Validate and sort DF candidate rows."""
    frame = candidates.copy()
    missing = [
        col for col in REQUIRED_CANDIDATE_COLUMNS if col not in frame.columns
    ]
    if missing:
        msg = 'DF candidate table is missing columns: {}'.format(
            ', '.join(missing)
        )
        raise ValueError(msg)

    # 数值列统一转成 float，避免 Excel 保存后类型变成字符串。
    numeric_cols = ['D_s', 'D_m', 'D_l', 'phi_s', 'phi_m', 'phi_l', 'E']
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')

    invalid = (
        frame[numeric_cols].isna().any(axis=1)
        | (frame[['D_s', 'D_m', 'D_l']] <= 0).any(axis=1)
        | (frame[['phi_s', 'phi_m', 'phi_l']] < 0).any(axis=1)
        | (frame['E'] < 0)
    )
    if invalid.any():
        bad_rows = ', '.join(map(str, frame.index[invalid].tolist()[:10]))
        msg = 'Invalid DF candidate rows at indices: {}'.format(bad_rows)
        raise ValueError(msg)

    # 配比可能因为小数保存有微小偏差，这里归一化回总和 1。
    phi_sum = frame[['phi_s', 'phi_m', 'phi_l']].sum(axis=1)
    frame[['phi_s', 'phi_m', 'phi_l']] = frame[
        ['phi_s', 'phi_m', 'phi_l']
    ].div(phi_sum, axis=0)

    if 'rank' in frame.columns:
        frame = frame.sort_values(['rank', 'E'])
    else:
        frame = frame.sort_values(['E', 'triplet_id'])

    return frame.reset_index(drop=True)


def merge_packing_data(candidates, packing_path):
    """Merge measured phi_max values from a low-cost packing CSV."""
    if packing_path is None:
        return candidates

    packing = pd.read_csv(packing_path)
    if 'phi_max' not in packing.columns:
        msg = 'Packing CSV must include a phi_max column.'
        raise ValueError(msg)

    # 推荐用 triplet_id 对齐；如果没有 triplet_id，就用三个粉体 ID 拼出来。
    if 'triplet_id' not in packing.columns:
        id_cols = ['small_id', 'medium_id', 'large_id']
        if all(col in packing.columns for col in id_cols):
            packing = packing.copy()
            packing['triplet_id'] = (
                packing['small_id'].astype(str)
                + '|'
                + packing['medium_id'].astype(str)
                + '|'
                + packing['large_id'].astype(str)
            )
        else:
            msg = 'Packing CSV needs triplet_id or small/medium/large IDs.'
            raise ValueError(msg)

    packing = packing[['triplet_id', 'phi_max']].copy()
    packing['phi_max'] = pd.to_numeric(packing['phi_max'], errors='coerce')
    if packing['phi_max'].isna().any():
        msg = 'Packing CSV contains non-numeric phi_max values.'
        raise ValueError(msg)

    merged = candidates.merge(
        packing.rename(columns={'phi_max': 'phi_max_measured'}),
        on='triplet_id',
        how='left',
    )
    return merged


def generate_lf_data(
    candidates,
    *,
    n_samples=0,
    seed=42,
    top_n_candidates=0,
    ratio_concentration=0.0,
    powders=None,
    df_exponent=0.37,
    grid_points=240,
    vf=0.30,
    vf_min=None,
    vf_max=None,
    k0=1.50,
    k_al2o3=30.0,
    shape_factor=1.5,
    phi_max_fixed=None,
    phi_max_upper=0.68,
    phi_max_lower=0.50,
    phi_max_error_scale=1.5,
    noise_frac=0.0,
):
    """Generate low-fidelity rows with the Lewis-Nielsen model."""
    rng = np.random.default_rng(seed)
    base = normalize_df_candidates(candidates)

    if top_n_candidates and top_n_candidates > 0:
        base = base.head(top_n_candidates)
    if base.empty:
        msg = 'No DF candidates are available for LF generation.'
        raise ValueError(msg)

    # n_samples <= 0 表示直接把每一行 DF 候选转换成一行 LF 数据。
    # 如果 n_samples 大于候选数，就按候选表重复抽样生成更多行。
    selected = _select_candidate_rows(base, n_samples, rng)

    # 如果要扰动三峰配比，必须能基于原始粉体库重新计算 E。
    if ratio_concentration > 0 and powders is None:
        msg = 'ratio_concentration > 0 requires the original --powders CSV.'
        raise ValueError(msg)

    ratios = _sample_ratios(
        selected[['phi_s', 'phi_m', 'phi_l']].to_numpy(dtype=float),
        ratio_concentration,
        rng,
    )
    selected = selected.copy().reset_index(drop=True)
    selected[['phi_s', 'phi_m', 'phi_l']] = ratios

    if powders is not None:
        selected['E'] = recompute_grading_errors(
            selected,
            powders,
            df_exponent=df_exponent,
            grid_points=grid_points,
        )
        selected['E_source'] = 'recomputed_from_powder_psd'
    else:
        selected['E_source'] = 'df_candidate'

    phi_max, phi_max_source = resolve_phi_max(
        selected,
        phi_max_fixed=phi_max_fixed,
        phi_max_upper=phi_max_upper,
        phi_max_lower=phi_max_lower,
        phi_max_error_scale=phi_max_error_scale,
    )
    V_f = sample_volume_fraction(len(selected), vf, vf_min, vf_max, rng)

    # Lewis-Nielsen 里的 V_f 不能超过 phi_max，否则公式会进入非物理区间。
    V_f, vf_clipped = clip_volume_fraction(V_f, phi_max)

    k_LF = lewis_nielsen_conductivity(
        k_matrix=k0,
        k_filler=k_al2o3,
        filler_fraction=V_f,
        phi_max=phi_max,
        shape_factor=shape_factor,
    )
    if noise_frac > 0:
        k_LF = k_LF * rng.normal(loc=1.0, scale=noise_frac, size=len(k_LF))

    output = selected.copy()
    output.insert(0, 'sample_id', _sample_ids(len(output)))
    output['source'] = 'LF_DF_Lewis_Nielsen'
    output['phi_max'] = phi_max
    output['phi_max_source'] = phi_max_source
    output['V_f'] = V_f
    output['vf_clipped'] = vf_clipped
    output['k0'] = k0
    output['k_matrix'] = k0
    output['k_al2o3'] = k_al2o3
    output['shape_factor'] = shape_factor
    output['k_LF'] = k_LF
    output['k'] = k_LF
    output['Delta_k_LF'] = k_LF - k0
    output['split'] = 'lf_pool'
    output['seed'] = seed
    output['notes'] = 'computed LF label; not measured HF data'

    return _order_output_columns(output)


def recompute_grading_errors(
    candidates,
    powders,
    *,
    df_exponent=0.37,
    grid_points=240,
):
    """Recompute E for candidate rows using the original powder PSD table."""
    powder_table = load_powder_table(powders) if isinstance(
        powders,
        (str, Path),
    ) else powders
    powder_table = powder_table.set_index('powder_id')
    errors = []

    for _, row in candidates.iterrows():
        small = powder_table.loc[str(row['small_id'])]
        medium = powder_table.loc[str(row['medium_id'])]
        large = powder_table.loc[str(row['large_id'])]

        d_min = float(row['d_min']) if 'd_min' in row else float(small['D10'])
        d_max = float(row['d_max']) if 'd_max' in row else float(large['D90'])
        diameters = np.geomspace(d_min, d_max, grid_points)
        target = dinger_funk_target(diameters, d_min, d_max, df_exponent)
        psds = np.vstack(
            [
                _lognormal_cdf_from_quantiles(diameters, small),
                _lognormal_cdf_from_quantiles(diameters, medium),
                _lognormal_cdf_from_quantiles(diameters, large),
            ]
        )
        ratio = np.asarray([[row['phi_s'], row['phi_m'], row['phi_l']]],
                           dtype=float)
        errors.append(
            _integrated_squared_error(ratio @ psds, target, diameters)[0]
        )

    return np.asarray(errors, dtype=float)


def resolve_phi_max(
    candidates,
    *,
    phi_max_fixed=None,
    phi_max_upper=0.68,
    phi_max_lower=0.50,
    phi_max_error_scale=1.5,
):
    """Resolve measured, fixed, or E-estimated maximum packing fraction."""
    if phi_max_fixed is not None:
        phi_max = np.full(len(candidates), float(phi_max_fixed))
        return _clip_phi_max(phi_max, phi_max_lower, phi_max_upper), 'fixed'

    if 'phi_max_measured' in candidates.columns:
        measured = pd.to_numeric(
            candidates['phi_max_measured'],
            errors='coerce',
        ).to_numpy(dtype=float)
        if np.isfinite(measured).any():
            estimated = estimate_phi_max_from_error(
                candidates['E'].to_numpy(dtype=float),
                upper=phi_max_upper,
                lower=phi_max_lower,
                error_scale=phi_max_error_scale,
            )
            phi_max = np.where(np.isfinite(measured), measured, estimated)
            source = np.where(
                np.isfinite(measured),
                'measured_packing',
                'estimated_from_E',
            )
            return _clip_phi_max(phi_max, phi_max_lower, phi_max_upper), source

    estimated = estimate_phi_max_from_error(
        candidates['E'].to_numpy(dtype=float),
        upper=phi_max_upper,
        lower=phi_max_lower,
        error_scale=phi_max_error_scale,
    )
    return estimated, 'estimated_from_E'


def save_training_tables(lf_data, features_output=None, labels_output=None):
    """Save optional PGNN feature and label CSV files."""
    if features_output is not None:
        features_output = Path(features_output)
        features_output.parent.mkdir(parents=True, exist_ok=True)
        lf_data[RAW_FEATURE_NAMES].to_csv(
            features_output,
            index=False,
            encoding='utf-8-sig',
        )

    if labels_output is not None:
        labels_output = Path(labels_output)
        labels_output.parent.mkdir(parents=True, exist_ok=True)
        lf_data[LABEL_NAMES].to_csv(
            labels_output,
            index=False,
            encoding='utf-8-sig',
        )


def _select_candidate_rows(candidates, n_samples, rng):
    if n_samples is None or n_samples <= 0:
        return candidates.copy()

    if n_samples <= len(candidates):
        return candidates.head(n_samples).copy()

    # 候选数少于目标 LF 行数时，循环使用优先级最高的候选行。
    indices = np.resize(np.arange(len(candidates)), n_samples)
    rng.shuffle(indices)
    return candidates.iloc[indices].copy()


def _sample_ratios(base_ratios, concentration, rng):
    if concentration <= 0:
        return base_ratios

    if concentration < 3:
        msg = 'ratio_concentration should be >= 3 for stable triplet ratios.'
        raise ValueError(msg)

    sampled = []
    for base in base_ratios:
        clipped = np.clip(base, 1.0e-6, None)
        clipped = clipped / clipped.sum()
        sampled.append(rng.dirichlet(clipped * concentration))

    return np.asarray(sampled, dtype=float)


def _sample_ids(n_rows):
    return ['LF_{:05d}'.format(i + 1) for i in range(n_rows)]


def _order_output_columns(output):
    first_cols = [
        'sample_id',
        'source',
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
        'E_source',
        'phi_max',
        'phi_max_source',
        'V_f',
        'vf_clipped',
        'k0',
        'k_matrix',
        'k_al2o3',
        'shape_factor',
        'k_LF',
        'k',
        'Delta_k_LF',
        'split',
        'seed',
        'notes',
    ]
    ordered = [col for col in first_cols if col in output.columns]
    remaining = [col for col in output.columns if col not in ordered]
    return output[ordered + remaining]


def _resolve_column(frame, candidates):
    # 把列名统一按小写匹配，减少 CSV 表头大小写差异造成的问题。
    lower_map = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        col = lower_map.get(candidate.lower())
        if col is not None:
            return col

    msg = 'Powder table needs one of these columns: {}'.format(
        ', '.join(candidates)
    )
    raise ValueError(msg)


def _find_optional_column(frame, candidates):
    try:
        return _resolve_column(frame, candidates)
    except ValueError:
        return None
