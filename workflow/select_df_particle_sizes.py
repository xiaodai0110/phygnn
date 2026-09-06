"""Select Al2O3 particle-size triplets with Dinger-Funk theory.

This is a thin CLI wrapper around :mod:`phygnn.bn_al2o3.data`. It searches
small/medium/large Al2O3 powder triplets and blend ratios, then ranks them by
the squared error between their mixed PSD and the Dinger-Funk target
cumulative PSD.

Expected powder CSV columns:
    powder_id,D10,D50,D90

The D10/D50/D90 values must use the same length unit, for example micrometers.
If no CSV is supplied, an embedded demo powder library is used so the script
can be run immediately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phygnn.bn_al2o3.data import (
    demo_powder_table,
    load_powder_table,
    select_df_triplets,
    write_template,
)
from phygnn.bn_al2o3.paths import data_path


def build_parser():
    """Build the command-line parser."""
    # 命令行参数集中控制粉体库路径、输出路径、搜索精度和筛选阈值。
    parser = argparse.ArgumentParser(
        description='Select Al2O3 particle-size triplets with Dinger-Funk PSD '
        'matching.'
    )
    parser.add_argument(
        '--powders',
        type=Path,
        default=None,
        help='CSV with columns powder_id,D10,D50,D90.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=data_path('df_particle_candidates.csv'),
        help='Output CSV path.',
    )
    parser.add_argument(
        '--write-template',
        type=Path,
        default=None,
        help='Write a powder-library template CSV and exit.',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Use embedded example powder data.',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=30,
        help='Number of ranked candidates to save. Use <=0 to save all.',
    )
    parser.add_argument(
        '--df-exponent',
        type=float,
        default=0.37,
        help='Dinger-Funk distribution modulus.',
    )
    parser.add_argument(
        '--ratio-step',
        type=float,
        default=0.05,
        help='Grid step for phi_s, phi_m, and phi_l.',
    )
    parser.add_argument(
        '--min-fraction',
        type=float,
        default=0.05,
        help='Minimum fraction allowed for each particle-size class.',
    )
    parser.add_argument(
        '--min-size-ratio',
        type=float,
        default=1.2,
        help='Minimum D50 ratio between adjacent size classes.',
    )
    parser.add_argument(
        '--grid-points',
        type=int,
        default=240,
        help='Number of PSD grid points for each triplet.',
    )
    parser.add_argument(
        '--d-min',
        type=float,
        default=None,
        help='Optional fixed minimum diameter for DF target.',
    )
    parser.add_argument(
        '--d-max',
        type=float,
        default=None,
        help='Optional fixed maximum diameter for DF target.',
    )
    parser.add_argument(
        '--max-error',
        type=float,
        default=None,
        help='Optional maximum grading error to keep.',
    )
    parser.add_argument(
        '--keep-all-ratios',
        action='store_true',
        help='Keep every ratio-grid candidate, not only the best one.',
    )
    return parser


def main(argv=None):
    """Run the particle-size screening CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # 如果还没有真实粉体库，可以先导出模板 CSV 再填写实测数据。
    if args.write_template is not None:
        write_template(args.write_template)
        print('Template written to {}'.format(args.write_template.resolve()))
        return 0

    # 不传 --powders 时默认跑 demo，避免点击运行时直接报错。
    if args.demo:
        powders = demo_powder_table()
        print('Using embedded demo powder data.')
    elif args.powders is None:
        powders = demo_powder_table()
        print('No --powders CSV supplied; using embedded demo powder data.')
    else:
        powders = load_powder_table(args.powders)

    results = select_df_triplets(
        powders,
        df_exponent=args.df_exponent,
        ratio_step=args.ratio_step,
        min_fraction=args.min_fraction,
        min_size_ratio=args.min_size_ratio,
        grid_points=args.grid_points,
        top_n=args.top_n,
        d_min=args.d_min,
        d_max=args.d_max,
        max_error=args.max_error,
        keep_all_ratios=args.keep_all_ratios,
    )

    # utf-8-sig 方便后续直接用 Excel 打开 CSV。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, encoding='utf-8-sig')

    # 终端只展示最关键字段，完整结果保存在输出 CSV 中。
    preview_cols = [
        'rank',
        'triplet_id',
        'D_s',
        'D_m',
        'D_l',
        'phi_s',
        'phi_m',
        'phi_l',
        'E',
    ]
    print('Saved {} candidates to {}'.format(len(results),
                                             args.output.resolve()))
    if len(results):
        print(results[preview_cols].head(10).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
