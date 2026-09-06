"""Train and predict BN/Al2O3 conductivity with the LF data pipeline.

This script connects the DF particle selector, the low-fidelity data
generator, and the PGNN model interface:

1. Load or generate low-fidelity rows.
2. Optionally load high-fidelity measured rows.
3. Train the BN/Al2O3 residual PGNN.
4. Predict total thermal conductivity k for candidate formulations.

Minimum training/prediction columns:
    D_s,D_m,D_l,phi_s,phi_m,E,k
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phygnn.bn_al2o3.constants import RAW_FEATURE_NAMES
from phygnn.bn_al2o3.data import (
    demo_powder_table,
    generate_lf_data,
    load_df_candidates,
    load_powder_table,
    merge_packing_data,
    select_df_triplets,
)
from phygnn.bn_al2o3.paths import data_path, models_path, outputs_path
from phygnn.model_interfaces.phygnn_model import PhygnnModel


def build_parser():
    """Build command-line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description='Generate LF data, train BN/Al2O3 PGNN, and predict k.'
    )

    # 数据入口：可以直接给 LF 数据，也可以给 DF 候选和粉体库现场生成 LF。
    parser.add_argument('--lf-data', type=Path, default=None)
    parser.add_argument(
        '--df-candidates',
        type=Path,
        default=data_path('df_particle_candidates.csv'),
    )
    parser.add_argument('--powders', type=Path, default=None)
    parser.add_argument('--packing-data', type=Path, default=None)
    parser.add_argument('--hf-data', type=Path, default=None)
    parser.add_argument('--hf-label-col', default=None)
    parser.add_argument('--predict-data', type=Path, default=None)

    # 输出入口：LF 表、模型目录和预测结果可以分别保存。
    parser.add_argument(
        '--lf-output',
        type=Path,
        default=data_path('lf_bn_al2o3_data.csv'),
    )
    parser.add_argument(
        '--model-output',
        type=Path,
        default=models_path('bn_al2o3_model'),
    )
    parser.add_argument(
        '--predictions-output',
        type=Path,
        default=outputs_path('bn_al2o3_predictions.csv'),
    )
    parser.add_argument('--no-save-model', action='store_true')
    parser.add_argument('--regenerate-lf', action='store_true')
    parser.add_argument('--demo', action='store_true')

    # LF 生成参数：和 generate_lf_bn_al2o3.py 保持一致。
    parser.add_argument('--n-lf-samples', type=int, default=0)
    parser.add_argument('--top-n-candidates', type=int, default=0)
    parser.add_argument('--ratio-concentration', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vf', type=float, default=0.30)
    parser.add_argument('--vf-min', type=float, default=None)
    parser.add_argument('--vf-max', type=float, default=None)
    parser.add_argument('--k-al2o3', type=float, default=30.0)
    parser.add_argument('--shape-factor', type=float, default=1.5)
    parser.add_argument('--phi-max-fixed', type=float, default=None)
    parser.add_argument('--phi-max-upper', type=float, default=0.68)
    parser.add_argument('--phi-max-lower', type=float, default=0.50)
    parser.add_argument('--phi-max-error-scale', type=float, default=1.5)
    parser.add_argument('--df-exponent', type=float, default=0.37)
    parser.add_argument('--grid-points', type=int, default=240)
    parser.add_argument('--noise-frac', type=float, default=0.0)

    # PGNN 训练参数：默认值偏向快速可运行；正式训练时建议加大 epoch。
    parser.add_argument('--baseline-mu', type=float, default=1.50)
    parser.add_argument('--baseline-stdev', type=float, default=0.08)
    parser.add_argument('--baseline-loss-weight', type=float, default=1.0)
    parser.add_argument('--stage1-epochs', type=int, default=1000)
    parser.add_argument('--stage2-epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--validation-split', type=float, default=0.2)
    parser.add_argument('--stage1-lr', type=float, default=1e-3)
    parser.add_argument('--stage2-lr', type=float, default=1e-4)
    parser.add_argument('--lf-weight', type=float, default=1.0)
    parser.add_argument('--hf-weight', type=float, default=10.0)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--early-stop-monitor', default='validation_loss')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
    parser.add_argument('--no-early-stop', action='store_true')

    # LF 数据集外部划分：默认 8:2，训练集进模型，测试集只做最终评估。
    parser.add_argument('--lf-test-size', type=float, default=0.2)
    parser.add_argument(
        '--lf-train-output',
        type=Path,
        default=data_path('lf_bn_al2o3_train.csv'),
    )
    parser.add_argument(
        '--lf-test-output',
        type=Path,
        default=data_path('lf_bn_al2o3_test.csv'),
    )
    parser.add_argument(
        '--lf-test-predictions-output',
        type=Path,
        default=outputs_path('lf_test_predictions.csv'),
    )
    parser.add_argument(
        '--history-output',
        type=Path,
        default=outputs_path('bn_al2o3_training_history.csv'),
    )
    parser.add_argument(
        '--loss-plot-output',
        type=Path,
        default=outputs_path('bn_al2o3_loss_curve.png'),
    )
    parser.add_argument('--no-loss-plot', action='store_true')
    return parser


def load_or_create_lf_data(args):
    """Load existing LF data or generate it from DF candidates."""
    if (
        args.lf_data is not None
        and args.lf_data.exists()
        and not args.regenerate_lf
    ):
        lf_data = pd.read_csv(args.lf_data)
        return normalize_training_table(lf_data, source_name='LF')

    candidates, powders = load_or_create_df_candidates(args)
    candidates = merge_packing_data(candidates, args.packing_data)

    lf_data = generate_lf_data(
        candidates,
        n_samples=args.n_lf_samples,
        seed=args.seed,
        top_n_candidates=args.top_n_candidates,
        ratio_concentration=args.ratio_concentration,
        powders=powders,
        df_exponent=args.df_exponent,
        grid_points=args.grid_points,
        vf=args.vf,
        vf_min=args.vf_min,
        vf_max=args.vf_max,
        k0=args.baseline_mu,
        k_al2o3=args.k_al2o3,
        shape_factor=args.shape_factor,
        phi_max_fixed=args.phi_max_fixed,
        phi_max_upper=args.phi_max_upper,
        phi_max_lower=args.phi_max_lower,
        phi_max_error_scale=args.phi_max_error_scale,
        noise_frac=args.noise_frac,
    )
    args.lf_output.parent.mkdir(parents=True, exist_ok=True)
    lf_data.to_csv(args.lf_output, index=False, encoding='utf-8-sig')
    return normalize_training_table(lf_data, source_name='LF')


def load_or_create_df_candidates(args):
    """Load DF candidates, or create them from a powder library."""
    if args.demo:
        powders = demo_powder_table()
        candidates = select_df_triplets(
            powders,
            top_n=max(args.n_lf_samples, 30),
            keep_all_ratios=True,
        )
        print('Using embedded demo DF candidates.')
        return candidates, powders

    if args.df_candidates.exists():
        candidates = load_df_candidates(args.df_candidates)
        powders = args.powders if args.powders is not None else None
        return candidates, powders

    if args.powders is not None and args.powders.exists():
        powders = load_powder_table(args.powders)
        candidates = select_df_triplets(
            powders,
            top_n=max(args.top_n_candidates, 30),
            keep_all_ratios=True,
        )
        return candidates, powders

    msg = (
        'No LF data or DF candidates were found. Provide --lf-data, run '
        'select_df_particle_sizes.py first, pass --powders, or use --demo.'
    )
    raise FileNotFoundError(msg)


def load_hf_data(path, label_col=None):
    """Load optional high-fidelity measured data."""
    if path is None:
        return None
    if not path.exists():
        msg = 'HF data file not found: {}'.format(path)
        raise FileNotFoundError(msg)

    frame = pd.read_csv(path)
    frame = normalize_training_table(frame, label_col=label_col,
                                     source_name='HF')

    # 文档要求重复实验不要拆分到不同集合，这里先按完全相同配方取均值。
    averaged = frame.groupby(RAW_FEATURE_NAMES, as_index=False).agg(
        k=('k', 'mean'),
        hf_repeat_count=('k', 'size'),
    )
    return averaged


def normalize_training_table(frame, label_col=None, source_name='data'):
    """Normalize a table into PGNN feature columns plus a k label."""
    data = frame.copy()
    missing_features = [
        col for col in RAW_FEATURE_NAMES if col not in data.columns
    ]
    if missing_features:
        msg = '{} table is missing feature columns: {}'.format(
            source_name,
            ', '.join(missing_features),
        )
        raise ValueError(msg)

    if label_col is None:
        if 'k' in data.columns:
            label_col = 'k'
        elif 'k_LF' in data.columns:
            label_col = 'k_LF'
        elif 'k_HF' in data.columns:
            label_col = 'k_HF'

    if label_col is None or label_col not in data.columns:
        msg = '{} table needs a k, k_LF, or k_HF label column.'.format(
            source_name
        )
        raise ValueError(msg)

    for col in RAW_FEATURE_NAMES:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['k'] = pd.to_numeric(data[label_col], errors='coerce')

    invalid = data[RAW_FEATURE_NAMES + ['k']].isna().any(axis=1)
    if invalid.any():
        bad_rows = ', '.join(map(str, data.index[invalid].tolist()[:10]))
        msg = '{} table has non-numeric rows: {}'.format(source_name, bad_rows)
        raise ValueError(msg)

    return data


def split_features_labels(training_table):
    """Split a normalized training table into PGNN feature and label frames."""
    features = training_table[RAW_FEATURE_NAMES].copy()
    labels = training_table[['k']].copy()
    return features, labels


def split_lf_train_test(lf_data, test_size=0.2, seed=42):
    """Split LF rows into train/test sets before PGNN training."""
    data = lf_data.copy().reset_index(drop=True)
    if test_size is None or test_size <= 0 or len(data) < 2:
        data['lf_split'] = 'train'
        return data, data.head(0).copy()

    if test_size < 1:
        n_test = int(round(len(data) * test_size))
    else:
        n_test = int(test_size)
    n_test = max(1, min(n_test, len(data) - 1))

    shuffled = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = shuffled.iloc[:n_test].copy()
    train = shuffled.iloc[n_test:].copy()
    train['lf_split'] = 'train'
    test['lf_split'] = 'test'
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_lf_splits(lf_train, lf_test, args):
    """Save train/test LF splits for reproducibility."""
    args.lf_train_output.parent.mkdir(parents=True, exist_ok=True)
    lf_train.to_csv(args.lf_train_output, index=False, encoding='utf-8-sig')

    args.lf_test_output.parent.mkdir(parents=True, exist_ok=True)
    lf_test.to_csv(args.lf_test_output, index=False, encoding='utf-8-sig')
    print(
        'Saved LF train/test split: {} train rows, {} test rows.'.format(
            len(lf_train),
            len(lf_test),
        )
    )


def train_model(lf_data, hf_data, args):
    """Train the BN/Al2O3 PGNN from LF data and optional HF data."""
    lf_features, lf_labels = split_features_labels(lf_data)
    model = PhygnnModel.build_bn_al2o3(
        baseline_mu=args.baseline_mu,
        baseline_stdev=args.baseline_stdev,
        baseline_loss_weight=args.baseline_loss_weight,
    )

    early_stop = not args.no_early_stop
    stop_kwargs = {
        'monitor': args.early_stop_monitor,
        'patience': args.patience,
        'min_delta': args.early_stop_min_delta,
    }

    if hf_data is None or hf_data.empty:
        print('No HF data supplied; training LF-only pretrain model.')
        diagnostics = model.train_multifidelity(
            lf_features=lf_features,
            lf_labels=lf_labels,
            loss_weights={
                'lf': args.lf_weight,
                'hf': 0.0,
                'baseline': args.baseline_loss_weight,
            },
            n_batch=None,
            batch_size=args.batch_size,
            n_epoch=args.stage1_epochs,
            validation_split=args.validation_split,
            learning_rate=args.stage1_lr,
            early_stop=early_stop,
            stop_kwargs=stop_kwargs,
            stage='lf_pretrain',
            return_diagnostics=True,
        )
        return model, diagnostics

    hf_features, hf_labels = split_features_labels(hf_data)
    print('Training two-stage LF+HF model.')
    diagnostics = model.train_two_stage_multifidelity(
        lf_features,
        lf_labels,
        hf_features,
        hf_labels,
        stage1_kwargs={
            'loss_weights': {
                'lf': args.lf_weight,
                'hf': 0.0,
                'baseline': args.baseline_loss_weight,
            },
            'n_batch': None,
            'batch_size': args.batch_size,
            'n_epoch': args.stage1_epochs,
            'validation_split': args.validation_split,
            'learning_rate': args.stage1_lr,
            'early_stop': early_stop,
            'stop_kwargs': stop_kwargs,
        },
        stage2_kwargs={
            'loss_weights': {
                'lf': args.lf_weight,
                'hf': args.hf_weight,
                'baseline': args.baseline_loss_weight,
            },
            'n_batch': None,
            'batch_size': args.batch_size,
            'n_epoch': args.stage2_epochs,
            'validation_split': args.validation_split,
            'learning_rate': args.stage2_lr,
            'early_stop': early_stop,
            'stop_kwargs': stop_kwargs,
        },
        return_diagnostics=True,
    )
    return model, diagnostics


def save_training_history(model, history_output):
    """Save model training history to CSV."""
    history = model.history.copy()
    history_output.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_output, index=True, encoding='utf-8-sig')
    print('Saved training history to {}'.format(history_output.resolve()))
    return history


def plot_loss_history(history, output_path):
    """Plot train_loss and val_loss curves."""
    plot_data = history.copy()
    plot_data = plot_data.reset_index().rename(columns={'epoch': 'epoch'})
    plot_data['training_loss'] = pd.to_numeric(
        plot_data['training_loss'],
        errors='coerce',
    )
    plot_data['validation_loss'] = pd.to_numeric(
        plot_data['validation_loss'],
        errors='coerce',
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(
        plot_data['epoch'],
        plot_data['training_loss'],
        label='train_loss',
        linewidth=2.0,
    )
    if plot_data['validation_loss'].notna().any():
        ax.plot(
            plot_data['epoch'],
            plot_data['validation_loss'],
            label='val_loss',
            linewidth=2.0,
        )

    if 'stage' in plot_data:
        _draw_stage_boundaries(ax, plot_data)

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('BN/Al2O3 PGNN loss history')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print('Saved loss plot to {}'.format(output_path.resolve()))


def evaluate_lf_test_set(model, lf_test, output_path):
    """Predict held-out LF test rows and calculate simple error metrics."""
    if lf_test.empty:
        return None

    output = predict_and_save(model, lf_test, output_path)
    mae = float(output['k_error'].abs().mean())
    rmse = float(np.sqrt((output['k_error'] ** 2).mean()))
    print('LF test MAE: {:.6f}; RMSE: {:.6f}'.format(mae, rmse))
    return {'lf_test_mae': mae, 'lf_test_rmse': rmse}


def load_prediction_table(args, lf_data):
    """Load prediction candidates or fall back to LF training rows."""
    if args.predict_data is None:
        return lf_data.copy()
    if not args.predict_data.exists():
        msg = 'Prediction data file not found: {}'.format(args.predict_data)
        raise FileNotFoundError(msg)

    prediction_data = pd.read_csv(args.predict_data)
    missing_features = [
        col for col in RAW_FEATURE_NAMES if col not in prediction_data
    ]
    if missing_features:
        msg = 'Prediction table is missing columns: {}'.format(
            ', '.join(missing_features)
        )
        raise ValueError(msg)

    return prediction_data


def predict_and_save(model, prediction_table, output_path):
    """Predict k and Delta k, then save the prediction table."""
    features = prediction_table[RAW_FEATURE_NAMES].copy()
    k_pred = model.predict_k(features, table=True).iloc[:, 0].to_numpy()
    delta_pred = (
        model.predict_delta(features, table=True).iloc[:, 0].to_numpy()
    )

    output = prediction_table.copy()
    output['k_pred'] = k_pred
    output['Delta_k_pred'] = delta_pred
    if 'k' in output.columns:
        output['k_error'] = output['k_pred'] - output['k']

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding='utf-8-sig')
    return output


def _draw_stage_boundaries(ax, plot_data):
    stage = plot_data['stage'].fillna('')
    changes = stage.ne(stage.shift()).to_numpy()
    for row_idx in np.where(changes)[0][1:]:
        epoch = plot_data.loc[row_idx, 'epoch']
        ax.axvline(epoch, color='0.6', linestyle='--', linewidth=1.0)


def main(argv=None):
    """Run the full LF-generation, training, and prediction pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    lf_data = load_or_create_lf_data(args)
    lf_train, lf_test = split_lf_train_test(
        lf_data,
        test_size=args.lf_test_size,
        seed=args.seed,
    )
    save_lf_splits(lf_train, lf_test, args)

    hf_data = load_hf_data(args.hf_data, label_col=args.hf_label_col)
    model, _diagnostics = train_model(lf_train, hf_data, args)
    history = save_training_history(model, args.history_output)
    if not args.no_loss_plot:
        plot_loss_history(history, args.loss_plot_output)
    evaluate_lf_test_set(model, lf_test, args.lf_test_predictions_output)

    if not args.no_save_model:
        model.save_model(str(args.model_output))
        print('Saved model to {}'.format(args.model_output.resolve()))

    prediction_table = load_prediction_table(args, lf_data)
    predictions = predict_and_save(
        model,
        prediction_table,
        args.predictions_output,
    )

    preview_cols = ['D_s', 'D_m', 'D_l', 'phi_s', 'phi_m', 'E', 'k_pred']
    print('Saved predictions to {}'.format(args.predictions_output.resolve()))
    print(predictions[preview_cols].head(10).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
