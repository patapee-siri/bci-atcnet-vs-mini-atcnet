"""Regenerate the polished comparison plots for README/results from the actual Kaggle run output.

The raw per-fold (45 rows/config) results_df is not available outside the Kaggle session -- only
this script's hardcoded numbers, which were read directly off the notebook's own printed output
(notebooks/02_atcnet_vs_mini_atcnet_augment_lrwarmup.ipynb, "Overall comparison" / "Per-subject
accuracy" / classification-report cells). Aggregate stats (mean/std across all 45 subject-fold
runs, exact) and per-subject means (9 values per config, each already averaged over 5 folds) are
used instead of a raw per-fold scatter -- every number below is copied verbatim from that output,
not estimated.

Run: python generate_plots.py   (writes *.png into this same results/ directory)
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

CONFIG_ORDER = ['ATCNet_full', 'Mini-ATCNet']
PALETTE = {'ATCNet_full': '#4C72B0', 'Mini-ATCNet': '#DD8452'}
CLASS_NAMES = ['Left', 'Right', 'Foot', 'Tongue']

PARAM_COUNTS = {'ATCNet_full': 115_172, 'Mini-ATCNet': 8_024}

# mean +/- std across all 45 subject-fold runs per config (n_epochs omitted: both configs ran the
# full 150-epoch budget on every single fold, std = 0)
AGGREGATE = {
    'ATCNet_full': dict(accuracy_mean=0.6683, accuracy_std=0.1657,
                         train_time_mean=159.8158, train_time_std=1.9845,
                         test_time_mean=110.1951, test_time_std=4.9170),
    'Mini-ATCNet': dict(accuracy_mean=0.5489, accuracy_std=0.1750,
                         train_time_mean=89.3498, train_time_std=1.6227,
                         test_time_mean=62.0352, test_time_std=3.8107),
}

# per-subject accuracy, each already averaged over that subject's 5 folds
PER_SUBJECT_ACC = pd.DataFrame({
    'subject': list(range(1, 10)),
    'ATCNet_full': [0.7569, 0.5734, 0.7846, 0.4826, 0.5242, 0.4238, 0.7985, 0.8367, 0.8335],
    'Mini-ATCNet': [0.5765, 0.4481, 0.7149, 0.3679, 0.3752, 0.3395, 0.6978, 0.7569, 0.6633],
})

# pooled classification report (all 9 subjects x 5 folds), per class
PER_CLASS_F1 = pd.DataFrame({
    'class': CLASS_NAMES * 2,
    'config': ['ATCNet_full'] * 4 + ['Mini-ATCNet'] * 4,
    'precision': [0.61, 0.65, 0.71, 0.72, 0.51, 0.55, 0.59, 0.56],
    'recall': [0.73, 0.70, 0.58, 0.66, 0.62, 0.59, 0.43, 0.56],
    'f1': [0.67, 0.68, 0.64, 0.68, 0.56, 0.57, 0.50, 0.56],
})

sns.set_theme(style='whitegrid', context='talk', font_scale=0.85)


def _bar_with_error(ax, means, stds, ylabel, title, fmt, pct=False):
    x = np.arange(len(CONFIG_ORDER))
    vals = [means[c] for c in CONFIG_ORDER]
    errs = [stds[c] for c in CONFIG_ORDER]
    bars = ax.bar(x, vals, yerr=errs, capsize=6, width=0.55,
                  color=[PALETTE[c] for c in CONFIG_ORDER], edgecolor='black', linewidth=0.8,
                  error_kw=dict(elinewidth=1.6, ecolor='#333333'), zorder=3)
    for b, v, e in zip(bars, vals, errs):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v + e), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_ORDER)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.0)
    ax.grid(axis='x', visible=False)
    sns.despine(ax=ax)


# -------------------- 1. accuracy / train time / inference time comparison --------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

acc_means = {c: AGGREGATE[c]['accuracy_mean'] for c in CONFIG_ORDER}
acc_stds = {c: AGGREGATE[c]['accuracy_std'] for c in CONFIG_ORDER}
_bar_with_error(axes[0], acc_means, acc_stds, 'Accuracy', 'Accuracy\n(mean +/- std, 45 subject-folds)',
                '{:.1%}', pct=True)

tt_means = {c: AGGREGATE[c]['train_time_mean'] for c in CONFIG_ORDER}
tt_stds = {c: AGGREGATE[c]['train_time_std'] for c in CONFIG_ORDER}
_bar_with_error(axes[1], tt_means, tt_stds, 'Seconds', 'Training time per fold\n(mean +/- std)', '{:.0f}s')

te_means = {c: AGGREGATE[c]['test_time_mean'] for c in CONFIG_ORDER}
te_stds = {c: AGGREGATE[c]['test_time_std'] for c in CONFIG_ORDER}
_bar_with_error(axes[2], te_means, te_stds, 'Milliseconds / trial', 'Inference time per trial\n(mean +/- std)',
                '{:.0f}ms')

fig.suptitle('ATCNet_full vs Mini-ATCNet -- Accuracy / Training Speed / Inference Speed',
             fontsize=16, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('atcnet_vs_mini_comparison.png', dpi=160, bbox_inches='tight')
plt.close(fig)


# -------------------- 2. per-subject accuracy grouped bar chart --------------------
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(PER_SUBJECT_ACC))
width = 0.35
for i, cfg in enumerate(CONFIG_ORDER):
    offset = (i - 0.5) * width
    bars = ax2.bar(x + offset, PER_SUBJECT_ACC[cfg], width=width, label=cfg,
                    color=PALETTE[cfg], edgecolor='black', linewidth=0.6, zorder=3)
    for b, v in zip(bars, PER_SUBJECT_ACC[cfg]):
        ax2.annotate(f'{v:.0%}', (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                     textcoords='offset points', ha='center', fontsize=8.5)

ax2.axhline(0.25, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Chance level (25%)')
ax2.set_xticks(x)
ax2.set_xticklabels([f'S{int(s):02d}' for s in PER_SUBJECT_ACC['subject']])
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax2.set_ylim(0, 1.0)
ax2.set_title('Per-subject accuracy (subject-dependent 5-fold CV)', fontsize=15, fontweight='bold', pad=12)
ax2.set_xlabel('Subject')
ax2.set_ylabel('Accuracy')
ax2.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11)
ax2.grid(axis='x', visible=False)
sns.despine(ax=ax2)
plt.tight_layout()
plt.savefig('atcnet_vs_mini_per_subject.png', dpi=160, bbox_inches='tight')
plt.close(fig2)


# -------------------- 3. parameter-count vs accuracy efficiency scatter --------------------
fig3, ax3 = plt.subplots(figsize=(7.5, 6))
for cfg in CONFIG_ORDER:
    acc_mean = AGGREGATE[cfg]['accuracy_mean']
    ax3.scatter(PARAM_COUNTS[cfg], acc_mean, s=280, color=PALETTE[cfg], edgecolor='black',
                linewidth=1.3, zorder=3)
    ax3.annotate(f"{cfg}\n{PARAM_COUNTS[cfg]:,} params\n{acc_mean:.1%} acc",
                 (PARAM_COUNTS[cfg], acc_mean), xytext=(14, 0), textcoords='offset points',
                 fontsize=11, va='center', fontweight='bold')
ax3.set_xscale('log')
ax3.set_xlim(3_000, 400_000)
ax3.set_ylim(0.3, 0.8)
ax3.set_xlabel('Trainable parameters (log scale)')
ax3.set_ylabel('Mean accuracy')
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax3.set_title('Parameter count vs accuracy trade-off', fontsize=15, fontweight='bold', pad=12)
ax3.grid(alpha=0.4)
sns.despine(ax=ax3)
plt.tight_layout()
plt.savefig('atcnet_vs_mini_efficiency.png', dpi=160, bbox_inches='tight')
plt.close(fig3)


# -------------------- 4. per-class F1 comparison --------------------
fig4, ax4 = plt.subplots(figsize=(9, 5.5))
sns.barplot(data=PER_CLASS_F1, x='class', y='f1', hue='config', order=CLASS_NAMES,
            hue_order=CONFIG_ORDER, palette=PALETTE, edgecolor='black', linewidth=0.6, ax=ax4,
            zorder=3)
for container in ax4.containers:
    ax4.bar_label(container, fmt='%.2f', fontsize=9, padding=2)
ax4.set_ylim(0, 0.85)
ax4.set_title('Per-class F1 score (pooled across all subjects/folds)', fontsize=15, fontweight='bold', pad=12)
ax4.set_xlabel('')
ax4.set_ylabel('F1 score')
ax4.legend(title='', frameon=False, loc='upper right')
ax4.grid(axis='x', visible=False)
sns.despine(ax=ax4)
plt.tight_layout()
plt.savefig('atcnet_vs_mini_per_class_f1.png', dpi=160, bbox_inches='tight')
plt.close(fig4)

print('wrote atcnet_vs_mini_comparison.png, atcnet_vs_mini_per_subject.png, '
      'atcnet_vs_mini_efficiency.png, atcnet_vs_mini_per_class_f1.png')
