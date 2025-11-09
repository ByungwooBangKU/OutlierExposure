#!/usr/bin/env python3
"""
WANDB 실험 결과 시각화 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 데이터 로드
df_self_oe = pd.read_csv('wandb_self_oe_results.csv')
df_completed = pd.read_csv('wandb_completed_results.csv')

print(f"Self-OE 실험: {len(df_self_oe)}개")
print(f"전체 완료 실험: {len(df_completed)}개")

# 출력 디렉토리 생성
output_dir = Path('paper_figures')
output_dir.mkdir(exist_ok=True)

# ===== 1. Baseline vs Self-OE 비교 (Bar Chart) =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df_baseline = df_completed[df_completed['use_self_attention_oe'] == False]
df_selfoe = df_completed[df_completed['use_self_attention_oe'] == True]

metrics = [
    ('auroc_mean', 'AUROC'),
    ('aupr_mean', 'AUPR'),
    ('fpr95_mean', 'FPR95 (↓)')
]

for idx, (metric, label) in enumerate(metrics):
    ax = axes[idx]

    baseline_mean = df_baseline[metric].mean()
    selfoe_mean = df_selfoe[metric].mean()
    selfoe_std = df_selfoe[metric].std()

    x = np.arange(2)
    bars = ax.bar(x, [baseline_mean, selfoe_mean], yerr=[0, selfoe_std],
                   capsize=5, alpha=0.8, color=['#E74C3C', '#3498DB'])

    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline\n(Standard)', 'Self-Attention\nOE'])
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f'{label} Comparison', fontsize=13, fontweight='bold')

    # 값 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

    # 개선률 표시
    if metric != 'fpr95_mean':
        improvement = ((selfoe_mean - baseline_mean) / baseline_mean * 100)
        ax.text(0.5, max(baseline_mean, selfoe_mean) * 0.95,
                f'개선: +{improvement:.2f}%',
                ha='center', fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        improvement = ((baseline_mean - selfoe_mean) / baseline_mean * 100)
        ax.text(0.5, max(baseline_mean, selfoe_mean) * 0.95,
                f'개선: +{improvement:.2f}%',
                ha='center', fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '01_baseline_vs_selfoe.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '01_baseline_vs_selfoe.pdf', bbox_inches='tight')
print("✓ Baseline vs Self-OE 비교 차트 저장")
plt.close()

# ===== 2. Masking Probability 효과 (Bar Chart) =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

mask_stats = df_self_oe.groupby('masking_probability').agg({
    'auroc_mean': ['mean', 'std'],
    'aupr_mean': ['mean', 'std'],
    'fpr95_mean': ['mean', 'std']
})

metrics_to_plot = [
    (('auroc_mean', 'mean'), ('auroc_mean', 'std'), 'AUROC'),
    (('aupr_mean', 'mean'), ('aupr_mean', 'std'), 'AUPR'),
    (('fpr95_mean', 'mean'), ('fpr95_mean', 'std'), 'FPR95 (↓)')
]

for idx, (mean_col, std_col, label) in enumerate(metrics_to_plot):
    ax = axes[idx]

    mask_probs = mask_stats.index.tolist()
    means = mask_stats[mean_col].values
    stds = mask_stats[std_col].values

    x = np.arange(len(mask_probs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color='#2ECC71')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{p:.2f}' for p in mask_probs])
    ax.set_xlabel('Masking Probability', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f'{label} by Masking Probability', fontsize=13, fontweight='bold')

    # 값 표시
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / '02_masking_probability_effect.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '02_masking_probability_effect.pdf', bbox_inches='tight')
print("✓ Masking Probability 효과 차트 저장")
plt.close()

# ===== 3. Attention Top-p 효과 (Bar Chart) =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

topp_stats = df_self_oe.groupby('attention_top_p').agg({
    'auroc_mean': ['mean', 'std'],
    'aupr_mean': ['mean', 'std'],
    'fpr95_mean': ['mean', 'std']
})

for idx, (mean_col, std_col, label) in enumerate(metrics_to_plot):
    ax = axes[idx]

    top_ps = topp_stats.index.tolist()
    means = topp_stats[mean_col].values
    stds = topp_stats[std_col].values

    x = np.arange(len(top_ps))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color='#9B59B6')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{p:.2f}' for p in top_ps])
    ax.set_xlabel('Attention Top-p', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f'{label} by Attention Top-p', fontsize=13, fontweight='bold')

    # 값 표시
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / '03_attention_topp_effect.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '03_attention_topp_effect.pdf', bbox_inches='tight')
print("✓ Attention Top-p 효과 차트 저장")
plt.close()

# ===== 4. Attention Filtering Method 비교 (Bar Chart) =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

filter_stats = df_self_oe.groupby('attention_filtering_method').agg({
    'auroc_mean': ['mean', 'std'],
    'aupr_mean': ['mean', 'std'],
    'fpr95_mean': ['mean', 'std']
}).sort_values(('auroc_mean', 'mean'), ascending=False)

for idx, (mean_col, std_col, label) in enumerate(metrics_to_plot):
    ax = axes[idx]

    methods = filter_stats.index.tolist()
    means = filter_stats[mean_col].values
    stds = filter_stats[std_col].values

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color='#E67E22')

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax.set_xlabel('Filtering Method', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f'{label} by Filtering Method', fontsize=13, fontweight='bold')

    # 최고값 강조
    best_idx = np.argmax(means) if label != 'FPR95 (↓)' else np.argmin(means)
    bars[best_idx].set_color('#E74C3C')
    bars[best_idx].set_alpha(1.0)

plt.tight_layout()
plt.savefig(output_dir / '04_filtering_method_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '04_filtering_method_comparison.pdf', bbox_inches='tight')
print("✓ Filtering Method 비교 차트 저장")
plt.close()

# ===== 5. Heatmap: Masking Prob x Attention Top-p (AUROC) =====
df_with_params = df_self_oe[
    df_self_oe['attention_top_p'].notna() &
    df_self_oe['masking_probability'].notna()
].copy()

pivot_auroc = df_with_params.pivot_table(
    values='auroc_mean',
    index='masking_probability',
    columns='attention_top_p',
    aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_auroc, annot=True, fmt='.4f', cmap='RdYlGn',
            cbar_kws={'label': 'AUROC'}, linewidths=0.5, ax=ax)
ax.set_title('AUROC Heatmap: Masking Probability × Attention Top-p',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Attention Top-p', fontsize=12)
ax.set_ylabel('Masking Probability', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / '05_heatmap_auroc.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '05_heatmap_auroc.pdf', bbox_inches='tight')
print("✓ AUROC Heatmap 저장")
plt.close()

# ===== 6. Line Chart: OOD Dataset별 성능 =====
fig, ax = plt.subplots(figsize=(12, 6))

filter_methods = df_self_oe['attention_filtering_method'].unique()
ood_datasets = ['ag_news', 'wikitext']

method_stats = df_self_oe.groupby('attention_filtering_method').agg({
    'auroc_ag_news': 'mean',
    'auroc_wikitext': 'mean'
})

x = np.arange(len(filter_methods))
width = 0.35

bars1 = ax.bar(x - width/2, method_stats['auroc_ag_news'], width,
               label='AG News', alpha=0.8, color='#3498DB')
bars2 = ax.bar(x + width/2, method_stats['auroc_wikitext'], width,
               label='WikiText', alpha=0.8, color='#E74C3C')

ax.set_xlabel('Filtering Method', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('AUROC by OOD Dataset and Filtering Method', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', '\n') for m in filter_methods], fontsize=9)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '06_ood_dataset_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '06_ood_dataset_comparison.pdf', bbox_inches='tight')
print("✓ OOD Dataset별 비교 차트 저장")
plt.close()

# ===== 7. Distribution Plot: AUROC Distribution =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline vs Self-OE
ax1 = axes[0]
ax1.hist(df_baseline['auroc_mean'], bins=10, alpha=0.6, label='Baseline', color='#E74C3C')
ax1.hist(df_selfoe['auroc_mean'], bins=20, alpha=0.6, label='Self-OE', color='#3498DB')
ax1.axvline(df_baseline['auroc_mean'].mean(), color='#E74C3C', linestyle='--', linewidth=2)
ax1.axvline(df_selfoe['auroc_mean'].mean(), color='#3498DB', linestyle='--', linewidth=2)
ax1.set_xlabel('AUROC', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('AUROC Distribution: Baseline vs Self-OE', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Masking Probability별
ax2 = axes[1]
for mask_prob in sorted(df_self_oe['masking_probability'].unique()):
    subset = df_self_oe[df_self_oe['masking_probability'] == mask_prob]
    ax2.hist(subset['auroc_mean'], bins=15, alpha=0.5, label=f'Mask={mask_prob:.2f}')
    ax2.axvline(subset['auroc_mean'].mean(), linestyle='--', linewidth=2)

ax2.set_xlabel('AUROC', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('AUROC Distribution by Masking Probability', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '07_auroc_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / '07_auroc_distribution.pdf', bbox_inches='tight')
print("✓ AUROC Distribution 차트 저장")
plt.close()

print(f"\n모든 시각화 자료가 '{output_dir}' 디렉토리에 저장되었습니다.")
print(f"총 7개의 차트 생성 완료!")
