"""Render AqSolDB figures using archived CSV/JSON records only.

Usage (from repository root): python scripts/make_result_figures.py

Does NOT refit models, generate predictions, or modify archived metrics.
"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'docs' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = '#256E93'
TEAL = '#168373'
AMBER = '#C97835'
INK = '#243448'
MUTED = '#66788A'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#AEB9C6',
    'svg.hashsalt': 'aqsoldb-scientific-figures-v1',
    'savefig.facecolor': 'white',
})


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def save(fig, stem):
    png = FIGURES / f'{stem}.png'
    svg = FIGURES / f'{stem}.svg'
    fig.savefig(png, dpi=240, bbox_inches='tight', facecolor='white')
    fig.savefig(svg, bbox_inches='tight', facecolor='white', metadata={'Date': None})
    # Matplotlib paths can include harmless trailing spaces rejected by git diff --check.
    lines = svg.read_text(encoding='utf-8').splitlines()
    svg.write_text('\n'.join(line.rstrip(' \t') for line in lines) + '\n', encoding='utf-8')
    plt.close(fig)
    print(f'Saved {png.relative_to(ROOT)} ({png.stat().st_size // 1024} KiB)')
    print(f'Saved {svg.relative_to(ROOT)} ({svg.stat().st_size // 1024} KiB)')


metrics = json.loads((RESULTS / 'final_test_metrics.json').read_text(encoding='utf-8'))
rows = read_csv(RESULTS / 'final_test_predictions.csv')
assert len(rows) == metrics['n_test'] == 1013, 'Unexpected test sample count'
obs = np.array([float(r['observed_logs']) for r in rows])
pred = np.array([float(r['predicted_logs']) for r in rows])
residual = np.array([float(r['residual']) for r in rows])
assert np.isfinite(obs).all() and np.isfinite(pred).all()
assert np.allclose(residual, obs - pred, atol=1e-5), 'Unexpected residual convention'

# Verify that the archived predictions are consistent with the archived report.
mae = np.mean(np.abs(pred - obs))
rmse = np.sqrt(np.mean((pred - obs)**2))
r2 = 1 - np.sum((pred - obs)**2) / np.sum((obs - obs.mean())**2)
assert abs(mae - metrics['test_mae']) < 1e-4
assert abs(rmse - metrics['test_rmse']) < 1e-4
assert abs(r2 - metrics['test_r2']) < 1e-4

# Figure 1: independent held-out regression performance.
fig, ax = plt.subplots(figsize=(7.6, 6.8))
ax.scatter(obs, pred, s=19, alpha=.43, color=BLUE, linewidths=0, rasterized=True)
lo = float(min(obs.min(), pred.min()) - .3)
hi = float(max(obs.max(), pred.max()) + .3)
ax.plot([lo, hi], [lo, hi], '--', color=AMBER, linewidth=1.8, label='Perfect prediction')
ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Observed LogS [log10(mol/L)]',
       ylabel='Predicted LogS [log10(mol/L)]', title='Held-out aqueous solubility prediction')
ax.set_aspect('equal', adjustable='box')
ax.grid(alpha=.15)
ax.legend(loc='lower right', frameon=False)
ax.text(.04, .96,
        f'N = {len(obs):,}\nR² = {metrics["test_r2"]:.3f}\n'
        f'RMSE = {metrics["test_rmse"]:.3f}\nMAE = {metrics["test_mae"]:.3f}',
        va='top', ha='left', transform=ax.transAxes, fontsize=11, color=INK,
        bbox={'boxstyle': 'round,pad=.6', 'facecolor': 'white', 'edgecolor': '#D9E1E8', 'alpha': .95})
fig.tight_layout()
save(fig, '01_observed_vs_predicted')

# Figure 2: 5 model families x 3 molecular representations, baseline CV.
cv_rows = read_csv(RESULTS / 'cv_summary.csv')
assert len(cv_rows) == 15 and len({(r['model'], r['feature_set']) for r in cv_rows}) == 15
models = ['Ridge Regression', 'SVR', 'Random Forest', 'XGBoost', 'LightGBM']
representations = ['descriptors', 'fingerprints', 'hybrid']
lookup = {(r['model'], r['feature_set']): float(r['mean_r2']) for r in cv_rows}
score = np.array([[lookup[(m, f)] for f in representations] for m in models])
fig, ax = plt.subplots(figsize=(8.6, 6.0))
im = ax.imshow(score, cmap='viridis', vmin=.40, vmax=.85, aspect='auto')
ax.set_xticks(range(3), ['RDKit descriptors', 'Morgan fingerprints', 'Hybrid'])
ax.set_yticks(range(5), models)
ax.tick_params(length=0, pad=8)
ax.set_title('Baseline model and representation comparison', pad=18)
for i in range(5):
    for j in range(3):
        value = score[i, j]
        rgb = im.cmap(im.norm(value))[:3]
        luminance = .2126*rgb[0] + .7152*rgb[1] + .0722*rgb[2]
        ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                color=INK if luminance > .48 else 'white', fontsize=12, fontweight='bold')
# Outline marks the family and features that were later selected after Optuna.
ax.add_patch(Rectangle((1.5, 2.5), 1, 1, fill=False, edgecolor='white', lw=3))
fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label='Mean 3-fold development CV R²')
fig.text(.5, .025, 'Cells show baseline CV only. Outlined: subsequently tuned and selected XGBoost + Hybrid.',
         ha='center', va='center', fontsize=9.4, color=MUTED)
fig.subplots_adjust(left=.20, bottom=.16, right=.94, top=.88)
save(fig, '02_baseline_cv_heatmap')

# Figure 3: diagnostic by observed-solubility quintile.
# Rank-based quintiles have near-equal counts even when LogS values repeat.
groups = np.array_split(np.argsort(obs, kind='stable'), 5)
mae_q = np.array([np.mean(np.abs(pred[idx] - obs[idx])) for idx in groups])
bias_q = np.array([np.mean(pred[idx] - obs[idx]) for idx in groups])
assert sum(len(idx) for idx in groups) == len(obs)
fig, ax = plt.subplots(figsize=(9.2, 5.9))
x = np.arange(5)
w = .36
bars_mae = ax.bar(x - w/2, mae_q, width=w, color=BLUE, label='MAE (absolute error)')
bars_bias = ax.bar(x + w/2, bias_q, width=w, color=AMBER, label='Mean signed error')
ax.axhline(0, color=INK, linewidth=1)
ax.set_xticks(x, ['Q1\nLeast soluble', 'Q2', 'Q3', 'Q4', 'Q5\nMost soluble'])
ax.set_ylabel('Prediction error (LogS)')
ax.set_title('Prediction error by observed solubility')
ax.set_ylim(min(-.92, float(bias_q.min()) - .2), max(1.46, float(mae_q.max()) + .3))
ax.grid(axis='y', alpha=.15)
ax.set_axisbelow(True)
for bars, vals, signed in ((bars_mae, mae_q, False), (bars_bias, bias_q, True)):
    for bar, value in zip(bars, vals):
        ax.annotate(f'{value:+.2f}' if signed else f'{value:.2f}',
                    (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 5 if value >= 0 else -7), textcoords='offset points',
                    ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
for i, idx in enumerate(groups):
    ax.text(i, 1.34, f'n={len(idx)}', ha='center', fontsize=9, color=MUTED)
ax.legend(loc='lower left', frameon=False, ncol=2)
fig.text(.5, .015, 'Quintiles use observed LogS. Signed error = predicted − observed; positive means overestimated solubility.',
         ha='center', va='bottom', fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, .055, 1, 1))
save(fig, '03_error_by_solubility')

print('\nPASS: 3 figures generated (PNG + SVG) from frozen result files only.')
print('No model training, tuning, or new predictions were performed.')
