"""Plot evaluation metrics for BN/Al2O3 PGNN predictions.

The script reads a prediction CSV that contains an observed conductivity
column and a predicted conductivity column. It then calculates common
regression metrics and saves a single PNG figure for quick inspection.
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from pathlib import Path

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phygnn.bn_al2o3.constants import (
    GROUP_COLUMN_CANDIDATES,
    OBSERVED_K_CANDIDATES,
    PREDICTED_K_CANDIDATES,
)
from phygnn.bn_al2o3.metrics import compute_regression_metrics
from phygnn.bn_al2o3.paths import outputs_path


def build_parser():
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot R2/RMSE/MAE metrics for BN/Al2O3 predictions.'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Prediction CSV. Defaults to lf_test_predictions.csv if found.',
    )
    parser.add_argument(
        '--label-col',
        default=None,
        help='Observed conductivity column. Auto-detected if omitted.',
    )
    parser.add_argument(
        '--pred-col',
        default=None,
        help='Predicted conductivity column. Auto-detected if omitted.',
    )
    parser.add_argument(
        '--group-col',
        default='auto',
        help='Optional grouping column for point color. Use none to disable.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=outputs_path('bn_al2o3_metrics.png'),
        help='Output PNG path.',
    )
    parser.add_argument(
        '--summary-output',
        type=Path,
        default=outputs_path('bn_al2o3_metrics_summary.csv'),
        help='Output CSV with metric values.',
    )
    parser.add_argument(
        '--title',
        default='BN/Al2O3 PGNN prediction metrics',
        help='Figure title.',
    )
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument(
        '--open',
        action='store_true',
        help='Open the generated PNG after saving it.',
    )
    return parser


def resolve_input_path(requested):
    """Resolve the input file path, using generated defaults if needed."""
    if requested is not None:
        if requested.exists():
            return requested
        msg = 'Input file not found: {}'.format(requested)
        raise FileNotFoundError(msg)

    defaults = [
        outputs_path('lf_test_predictions.csv'),
        outputs_path('bn_al2o3_predictions.csv'),
    ]
    for default_path in defaults:
        if default_path.exists():
            return default_path

    msg = (
        'No default prediction CSV found. Provide --input, or run '
        'train_bn_al2o3.py first.'
    )
    raise FileNotFoundError(msg)


def choose_column(frame, requested, candidates, role):
    """Choose a column either from user input or known alternatives."""
    if requested is not None:
        if requested in frame.columns:
            return requested
        msg = '{} column not found: {}'.format(role, requested)
        raise ValueError(msg)

    for column in candidates:
        if column in frame.columns:
            return column

    msg = '{} column not found. Tried: {}'.format(role, ', '.join(candidates))
    raise ValueError(msg)


def choose_group_column(frame, requested):
    """Choose an optional group column for coloring points."""
    if requested is None:
        return None
    if requested.lower() in {'', 'none', 'false', 'off'}:
        return None
    if requested != 'auto':
        if requested not in frame.columns:
            msg = 'Group column not found: {}'.format(requested)
            raise ValueError(msg)
        return requested

    for column in GROUP_COLUMN_CANDIDATES:
        if column not in frame.columns:
            continue
        n_groups = frame[column].nunique(dropna=True)
        if 1 < n_groups <= 8:
            return column
    return None


def load_evaluation_table(input_path, label_col=None, pred_col=None):
    """Load prediction rows and normalize observed/predicted columns."""
    raw = pd.read_csv(input_path)
    label_col = choose_column(
        raw,
        label_col,
        OBSERVED_K_CANDIDATES,
        'Observed conductivity',
    )
    pred_col = choose_column(
        raw,
        pred_col,
        PREDICTED_K_CANDIDATES,
        'Predicted conductivity',
    )

    data = raw.copy()
    data['_y_true'] = pd.to_numeric(data[label_col], errors='coerce')
    data['_y_pred'] = pd.to_numeric(data[pred_col], errors='coerce')
    data = data.dropna(subset=['_y_true', '_y_pred']).reset_index(drop=True)
    if data.empty:
        msg = 'No valid numeric rows were found for evaluation.'
        raise ValueError(msg)

    data['_error'] = data['_y_pred'] - data['_y_true']
    data['_abs_error'] = data['_error'].abs()
    return data, label_col, pred_col


def save_metrics_summary(metrics, output_path):
    """Save metrics to a small CSV table."""
    rows = [
        {'metric': metric, 'value': value}
        for metric, value in metrics.items()
    ]
    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding='utf-8-sig')


def plot_metrics_figure(
    data,
    metrics,
    output_path,
    title,
    dpi=180,
    group_col=None,
):
    """Create and save the evaluation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=dpi)
    fig.suptitle(title, fontsize=15, fontweight='bold')

    plot_parity(axes[0, 0], data, metrics, group_col)
    plot_residuals(axes[0, 1], data, group_col)
    plot_error_histogram(axes[1, 0], data)
    plot_metric_table(axes[1, 1], metrics)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_parity(ax, data, metrics, group_col=None):
    """Plot predicted conductivity against observed conductivity."""
    plot_grouped_scatter(
        ax,
        data,
        x_col='_y_true',
        y_col='_y_pred',
        group_col=group_col,
    )
    low, high = get_equal_axis_limits(data['_y_true'], data['_y_pred'])
    ax.plot([low, high], [low, high], color='0.2', linewidth=1.4)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel('Observed k')
    ax.set_ylabel('Predicted k')
    ax.set_title('Prediction parity')
    ax.grid(True, alpha=0.22)

    label = 'R$^2$ = {}\nRMSE = {}'.format(
        format_metric(metrics['r2']),
        format_metric(metrics['rmse']),
    )
    ax.text(
        0.04,
        0.96,
        label,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox={'boxstyle': 'round,pad=0.35', 'fc': 'white', 'alpha': 0.82},
    )
    add_legend_if_needed(ax, group_col)


def plot_residuals(ax, data, group_col=None):
    """Plot prediction residuals against predicted values."""
    plot_grouped_scatter(
        ax,
        data,
        x_col='_y_pred',
        y_col='_error',
        group_col=group_col,
    )
    ax.axhline(0.0, color='0.2', linewidth=1.3)
    ax.set_xlabel('Predicted k')
    ax.set_ylabel('Prediction error')
    ax.set_title('Residuals')
    ax.grid(True, alpha=0.22)
    add_legend_if_needed(ax, group_col)


def plot_error_histogram(ax, data):
    """Plot the prediction error distribution."""
    errors = data['_error'].to_numpy(dtype=float)
    bins = min(24, max(6, int(np.sqrt(len(errors)))))
    ax.hist(errors, bins=bins, color='#6B8E23', alpha=0.82, edgecolor='white')
    ax.axvline(0.0, color='0.2', linewidth=1.3)
    ax.axvline(np.mean(errors), color='#B24745', linestyle='--', linewidth=1.4)
    ax.set_xlabel('Prediction error')
    ax.set_ylabel('Count')
    ax.set_title('Error distribution')
    ax.grid(True, axis='y', alpha=0.22)


def plot_metric_table(ax, metrics):
    """Draw a compact metrics table."""
    ax.axis('off')
    rows = [
        ['Samples', '{:.0f}'.format(metrics['n'])],
        ['R2', format_metric(metrics['r2'])],
        ['RMSE', format_metric(metrics['rmse'])],
        ['MAE', format_metric(metrics['mae'])],
        ['MAPE', '{}%'.format(format_metric(metrics['mape_percent']))],
        ['Bias', format_metric(metrics['bias'])],
        ['Max AE', format_metric(metrics['max_abs_error'])],
        ['Mean observed', format_metric(metrics['mean_observed'])],
        ['Mean predicted', format_metric(metrics['mean_predicted'])],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor('0.82')
        if row == 0:
            cell.set_facecolor('#20343B')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#F6F7F7' if row % 2 == 0 else 'white')
    ax.set_title('Metric summary')


def plot_grouped_scatter(ax, data, x_col, y_col, group_col=None):
    """Plot a scatter chart with optional grouping."""
    if group_col is None:
        ax.scatter(
            data[x_col],
            data[y_col],
            s=42,
            alpha=0.78,
            color='#2F6B8A',
            edgecolor='white',
            linewidth=0.5,
        )
        return

    for group_name, group_data in data.groupby(group_col, dropna=False):
        ax.scatter(
            group_data[x_col],
            group_data[y_col],
            s=42,
            alpha=0.78,
            label=str(group_name),
            edgecolor='white',
            linewidth=0.5,
        )


def add_legend_if_needed(ax, group_col=None):
    """Add a legend when a grouping column is active."""
    if group_col is not None:
        ax.legend(title=group_col, fontsize=8, title_fontsize=8)


def get_equal_axis_limits(x_values, y_values):
    """Return shared axis limits for parity plots."""
    values = np.concatenate([
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
    ])
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 0.1)
    else:
        padding = (high - low) * 0.08
    return low - padding, high + padding


def format_metric(value):
    """Format metric values for plots and terminal output."""
    if value is None or not np.isfinite(value):
        return 'NA'
    abs_value = abs(value)
    if abs_value >= 100:
        return '{:.2f}'.format(value)
    if abs_value >= 10:
        return '{:.3f}'.format(value)
    return '{:.4f}'.format(value)


def print_metrics(metrics, input_path, output_path, summary_output):
    """Print a concise terminal summary."""
    print('Loaded evaluation data: {}'.format(input_path.resolve()))
    print('Saved metrics figure: {}'.format(output_path.resolve()))
    print('Saved metrics summary: {}'.format(summary_output.resolve()))
    for metric in [
        'n',
        'r2',
        'rmse',
        'mae',
        'mape_percent',
        'bias',
        'max_abs_error',
    ]:
        print('{}: {}'.format(metric, format_metric(metrics[metric])))


def main(argv=None):
    """Run the metrics plotting workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = resolve_input_path(args.input)
    data, _label_col, _pred_col = load_evaluation_table(
        input_path,
        label_col=args.label_col,
        pred_col=args.pred_col,
    )
    group_col = choose_group_column(data, args.group_col)
    metrics = compute_regression_metrics(data['_y_true'], data['_y_pred'])

    save_metrics_summary(metrics, args.summary_output)
    plot_metrics_figure(
        data,
        metrics,
        args.output,
        title=args.title,
        dpi=args.dpi,
        group_col=group_col,
    )
    print_metrics(metrics, input_path, args.output, args.summary_output)

    if args.open:
        webbrowser.open(args.output.resolve().as_uri())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
