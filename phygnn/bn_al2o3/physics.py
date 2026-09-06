# -*- coding: utf-8 -*-
"""Pure physics / math helpers for the BN/Al2O3 workflow.

This module has no pandas or filesystem dependencies so the equations can be
tested and reused independently of the data pipeline.
"""
from math import erf, sqrt

import numpy as np

# 标准正态分布下累计概率为 90% 的 z 值。
# 用 D10/D50/D90 反推对数正态 PSD 宽度时会用到它。
Z_90 = 1.2815515655446004
SQRT_TWO = sqrt(2.0)


def dinger_funk_target(diameters, d_min, d_max, df_exponent=0.37):
    """Calculate the Dinger-Funk cumulative passing fraction."""
    # Dinger-Funk 目标曲线：
    # P(D) = (D^q - D_min^q) / (D_max^q - D_min^q)
    # q 常取 0.37，也可以用 --df-exponent 调整。
    if df_exponent <= 0:
        msg = 'df_exponent must be greater than zero.'
        raise ValueError(msg)
    if d_min <= 0 or d_max <= d_min:
        msg = 'd_min must be positive and d_max must be greater than d_min.'
        raise ValueError(msg)

    diameters = np.asarray(diameters, dtype=float)
    # 超出目标粒径范围的点先夹到边界，避免累计率小于 0 或大于 1。
    bounded = np.clip(diameters, d_min, d_max)
    numerator = np.power(bounded, df_exponent) - d_min**df_exponent
    denominator = d_max**df_exponent - d_min**df_exponent
    return np.clip(numerator / denominator, 0.0, 1.0)


def make_ratio_grid(step=0.05, min_fraction=0.05):
    """Build a deterministic ternary ratio grid."""
    # 三峰配比满足 phi_s + phi_m + phi_l = 1。
    # 这里只枚举 phi_s 和 phi_m，phi_l 由总和约束自动得到。
    if step <= 0 or step >= 1:
        msg = 'step must be greater than 0 and less than 1.'
        raise ValueError(msg)
    if min_fraction < 0 or min_fraction >= 1.0 / 3.0:
        msg = 'min_fraction must be >= 0 and less than 1/3.'
        raise ValueError(msg)

    tol = step * 1.0e-6
    values = np.arange(min_fraction, 1.0 + 0.5 * step, step)
    rows = []
    for phi_s in values:
        for phi_m in values:
            phi_l = 1.0 - phi_s - phi_m
            if phi_l >= min_fraction - tol:
                rows.append((round(phi_s, 10), round(phi_m, 10),
                             round(phi_l, 10)))

    if not rows:
        msg = 'No valid ratios were generated; relax step or min_fraction.'
        raise ValueError(msg)

    # 去重后按配比排序，保证每次运行的搜索顺序和结果可复现。
    ratios = np.unique(np.asarray(rows, dtype=float), axis=0)
    return ratios[np.lexsort((ratios[:, 2], ratios[:, 1], ratios[:, 0]))]


def _lognormal_cdf_from_quantiles(diameters, powder):
    # 这里把单一粉体的 PSD 近似为对数正态分布。
    # D50 是中位数，D10/D90 用来反推分布宽度 sigma。
    log_d10 = np.log(float(powder['D10']))
    log_d50 = np.log(float(powder['D50']))
    log_d90 = np.log(float(powder['D90']))
    sigma = (log_d90 - log_d10) / (2.0 * Z_90)
    if sigma <= 0:
        msg = 'Powder {} has invalid PSD quantiles.'.format(
            powder['powder_id']
        )
        raise ValueError(msg)

    z_score = (np.log(diameters) - log_d50) / sigma
    erf_vec = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(z_score / SQRT_TWO))


def _integrated_squared_error(mixes, target, diameters):
    # E = mean integral (P_mix - P_DF)^2 d(log D)。
    # 用 log(D) 积分可以让不同数量级粒径段的权重更均衡。
    x_axis = np.log(diameters)
    delta = (mixes - target[None, :]) ** 2
    trapz = getattr(np, 'trapezoid', None)
    if trapz is None:
        trapz = getattr(np, 'trapz')
    error = trapz(delta, x=x_axis, axis=1)
    return error / (x_axis[-1] - x_axis[0])


def lewis_nielsen_conductivity(
    *,
    k_matrix,
    k_filler,
    filler_fraction,
    phi_max,
    shape_factor=1.5,
):
    """Calculate effective conductivity with the Lewis-Nielsen equation."""
    if k_matrix <= 0 or k_filler <= 0:
        msg = 'k_matrix and k_filler must be positive.'
        raise ValueError(msg)
    if shape_factor <= 0:
        msg = 'shape_factor must be positive.'
        raise ValueError(msg)

    filler_fraction = np.asarray(filler_fraction, dtype=float)
    phi_max = np.asarray(phi_max, dtype=float)
    if (filler_fraction < 0).any():
        msg = 'filler_fraction must be non-negative.'
        raise ValueError(msg)
    if (phi_max <= 0).any():
        msg = 'phi_max must be positive.'
        raise ValueError(msg)

    # Lewis-Nielsen:
    # k/k_m = (1 + A B V_f) / (1 - B psi V_f)
    # B = (k_f/k_m - 1) / (k_f/k_m + A)
    # psi = 1 + ((1 - phi_max) / phi_max^2) V_f
    conductivity_ratio = k_filler / k_matrix
    B = (conductivity_ratio - 1.0) / (conductivity_ratio + shape_factor)
    psi = 1.0 + ((1.0 - phi_max) / phi_max**2) * filler_fraction
    denominator = 1.0 - B * psi * filler_fraction
    if (denominator <= 0).any():
        msg = 'Lewis-Nielsen denominator became non-positive.'
        raise ValueError(msg)

    numerator = 1.0 + shape_factor * B * filler_fraction
    return k_matrix * numerator / denominator


def estimate_phi_max_from_error(
    E,
    *,
    upper=0.68,
    lower=0.50,
    error_scale=1.5,
):
    """Estimate phi_max from grading error when packing data is unavailable."""
    # 这是无实测装填数据时的工程近似：
    # E 越小，说明越接近 DF 理想级配，phi_max 越接近 upper。
    # E 越大，说明级配偏离理想曲线，phi_max 被下调。
    E = np.asarray(E, dtype=float)
    phi_max = upper - error_scale * np.sqrt(np.maximum(E, 0.0))
    return _clip_phi_max(phi_max, lower, upper)


def _clip_phi_max(phi_max, lower, upper):
    if lower <= 0 or upper <= lower:
        msg = 'Need 0 < phi_max_lower < phi_max_upper.'
        raise ValueError(msg)
    return np.clip(np.asarray(phi_max, dtype=float), lower, upper)


def sample_volume_fraction(n_rows, vf, vf_min, vf_max, rng):
    """Sample or fix total Al2O3 filler volume fraction."""
    if vf_min is None and vf_max is None:
        return np.full(n_rows, vf, dtype=float)
    if vf_min is None or vf_max is None or vf_max < vf_min:
        msg = 'vf_min and vf_max must both be set, with vf_max >= vf_min.'
        raise ValueError(msg)
    return rng.uniform(vf_min, vf_max, size=n_rows)


def clip_volume_fraction(V_f, phi_max, guard=0.98):
    """Clip V_f below phi_max so Lewis-Nielsen stays physical."""
    limit = guard * np.asarray(phi_max, dtype=float)
    clipped = V_f > limit
    return np.minimum(V_f, limit), clipped
