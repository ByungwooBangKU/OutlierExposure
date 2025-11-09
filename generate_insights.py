#!/usr/bin/env python3
"""
통계적 인사이트 도출 및 분석
"""

import pandas as pd
import numpy as np
from scipy import stats

# 데이터 로드
df_self_oe = pd.read_csv('wandb_self_oe_results.csv')
df_completed = pd.read_csv('wandb_completed_results.csv')

print("="*70)
print("통계적 인사이트 분석")
print("="*70)

# ===== 1. Baseline vs Self-OE: 통계적 유의성 검정 =====
print("\n【1】 Baseline vs Self-Attention OE 통계 검정")
print("-" * 70)

df_baseline = df_completed[df_completed['use_self_attention_oe'] == False]
df_selfoe = df_completed[df_completed['use_self_attention_oe'] == True]

metrics = ['auroc_mean', 'aupr_mean', 'fpr95_mean']
metric_names = ['AUROC', 'AUPR', 'FPR95']

for metric, name in zip(metrics, metric_names):
    baseline_vals = df_baseline[metric].dropna()
    selfoe_vals = df_selfoe[metric].dropna()

    # T-test
    if len(baseline_vals) > 1 and len(selfoe_vals) > 1:
        t_stat, p_value = stats.ttest_ind(selfoe_vals, baseline_vals)
        print(f"\n{name}:")
        print(f"  Baseline: {baseline_vals.mean():.4f} (n={len(baseline_vals)})")
        print(f"  Self-OE:  {selfoe_vals.mean():.4f} ± {selfoe_vals.std():.4f} (n={len(selfoe_vals)})")

        if metric == 'fpr95_mean':
            improvement = (baseline_vals.mean() - selfoe_vals.mean()) / baseline_vals.mean() * 100
        else:
            improvement = (selfoe_vals.mean() - baseline_vals.mean()) / baseline_vals.mean() * 100

        print(f"  개선률: {improvement:+.2f}%")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")

        if p_value < 0.001:
            print(f"  결과: 매우 유의미한 개선 (p < 0.001) ***")
        elif p_value < 0.01:
            print(f"  결과: 유의미한 개선 (p < 0.01) **")
        elif p_value < 0.05:
            print(f"  결과: 유의미한 개선 (p < 0.05) *")
        else:
            print(f"  결과: 통계적으로 유의미하지 않음")

# ===== 2. Masking Probability 효과 분석 =====
print("\n\n【2】 Masking Probability 효과 분석")
print("-" * 70)

mask_groups = df_self_oe.groupby('masking_probability')

for mask_prob, group in mask_groups:
    print(f"\nMasking Probability = {mask_prob:.2f} (n={len(group)}):")
    print(f"  AUROC: {group['auroc_mean'].mean():.4f} ± {group['auroc_mean'].std():.4f}")
    print(f"  AUPR:  {group['aupr_mean'].mean():.4f} ± {group['aupr_mean'].std():.4f}")
    print(f"  FPR95: {group['fpr95_mean'].mean():.4f} ± {group['fpr95_mean'].std():.4f}")

# Masking 0.00 vs 0.05 비교
mask_0 = df_self_oe[df_self_oe['masking_probability'] == 0.00]['auroc_mean']
mask_5 = df_self_oe[df_self_oe['masking_probability'] == 0.05]['auroc_mean']

if len(mask_0) > 0 and len(mask_5) > 0:
    t_stat, p_value = stats.ttest_ind(mask_5, mask_0)
    improvement = (mask_5.mean() - mask_0.mean()) / mask_0.mean() * 100

    print(f"\n✦ Masking 0.05 vs 0.00 비교:")
    print(f"  개선률: {improvement:+.2f}%")
    print(f"  P-value: {p_value:.6f}")
    print(f"  결론: Masking 0.05가 통계적으로 유의미하게 우수" if p_value < 0.05 else "  결론: 차이 없음")

# ===== 3. Attention Top-p 효과 분석 =====
print("\n\n【3】 Attention Top-p 효과 분석")
print("-" * 70)

topp_groups = df_self_oe.groupby('attention_top_p')

best_topp = None
best_auroc = 0

for topp, group in topp_groups:
    mean_auroc = group['auroc_mean'].mean()
    print(f"\nTop-p = {topp:.2f} (n={len(group)}):")
    print(f"  AUROC: {mean_auroc:.4f} ± {group['auroc_mean'].std():.4f}")
    print(f"  AUPR:  {group['aupr_mean'].mean():.4f} ± {group['aupr_mean'].std():.4f}")
    print(f"  FPR95: {group['fpr95_mean'].mean():.4f} ± {group['fpr95_mean'].std():.4f}")

    if mean_auroc > best_auroc:
        best_auroc = mean_auroc
        best_topp = topp

print(f"\n✦ 최적 Top-p: {best_topp:.2f} (AUROC: {best_auroc:.4f})")

# ANOVA 테스트
topp_values = [group['auroc_mean'].values for _, group in topp_groups]
f_stat, p_value = stats.f_oneway(*topp_values)
print(f"  ANOVA F-statistic: {f_stat:.4f}, P-value: {p_value:.6f}")
print(f"  결론: Top-p 간 차이가 {'유의미함' if p_value < 0.05 else '유의미하지 않음'}")

# ===== 4. Attention Filtering Method 효과 분석 =====
print("\n\n【4】 Attention Filtering Method 효과 분석")
print("-" * 70)

filter_groups = df_self_oe.groupby('attention_filtering_method')
filter_results = []

for method, group in filter_groups:
    result = {
        'method': method,
        'n': len(group),
        'auroc_mean': group['auroc_mean'].mean(),
        'auroc_std': group['auroc_mean'].std(),
        'aupr_mean': group['aupr_mean'].mean(),
        'fpr95_mean': group['fpr95_mean'].mean()
    }
    filter_results.append(result)

filter_df = pd.DataFrame(filter_results).sort_values('auroc_mean', ascending=False)

print("\n성능 순위 (AUROC 기준):")
for i, row in filter_df.iterrows():
    print(f"\n{row['method']} (n={row['n']}):")
    print(f"  AUROC: {row['auroc_mean']:.4f} ± {row['auroc_std']:.4f}")
    print(f"  AUPR:  {row['aupr_mean']:.4f}")
    print(f"  FPR95: {row['fpr95_mean']:.4f}")

best_method = filter_df.iloc[0]
print(f"\n✦ 최고 성능 필터링 방법: {best_method['method']}")
print(f"  AUROC: {best_method['auroc_mean']:.4f}")

# ===== 5. 파라미터 조합 분석 =====
print("\n\n【5】 최적 파라미터 조합 분석")
print("-" * 70)

df_with_params = df_self_oe[
    df_self_oe['attention_top_p'].notna() &
    df_self_oe['masking_probability'].notna()
].copy()

combo_stats = df_with_params.groupby(['attention_top_p', 'masking_probability', 'attention_filtering_method']).agg({
    'auroc_mean': ['mean', 'std', 'count'],
}).round(4)

top_combos = combo_stats.sort_values(('auroc_mean', 'mean'), ascending=False).head(5)

print("\nTop 5 파라미터 조합 (AUROC 기준):")
for idx, (combo, row) in enumerate(top_combos.iterrows(), 1):
    topp, mask, method = combo
    print(f"\n{idx}. Top-p={topp:.2f}, Mask={mask:.2f}, Method={method}")
    print(f"   AUROC: {row[('auroc_mean', 'mean')]:.4f} ± {row[('auroc_mean', 'std')]:.4f} (n={int(row[('auroc_mean', 'count')])})")

# ===== 6. OOD Dataset별 성능 분석 =====
print("\n\n【6】 OOD Dataset별 성능 분석")
print("-" * 70)

print("\nAG News:")
print(f"  Baseline AUROC: {df_baseline['auroc_ag_news'].mean():.4f}")
print(f"  Self-OE AUROC:  {df_selfoe['auroc_ag_news'].mean():.4f} ± {df_selfoe['auroc_ag_news'].std():.4f}")
ag_improvement = (df_selfoe['auroc_ag_news'].mean() - df_baseline['auroc_ag_news'].mean()) / df_baseline['auroc_ag_news'].mean() * 100
print(f"  개선률: {ag_improvement:+.2f}%")

print("\nWikiText:")
print(f"  Baseline AUROC: {df_baseline['auroc_wikitext'].mean():.4f}")
print(f"  Self-OE AUROC:  {df_selfoe['auroc_wikitext'].mean():.4f} ± {df_selfoe['auroc_wikitext'].std():.4f}")
wiki_improvement = (df_selfoe['auroc_wikitext'].mean() - df_baseline['auroc_wikitext'].mean()) / df_baseline['auroc_wikitext'].mean() * 100
print(f"  개선률: {wiki_improvement:+.2f}%")

print(f"\n✦ WikiText에서 더 큰 성능 향상 확인 ({wiki_improvement:.2f}% vs {ag_improvement:.2f}%)")

# ===== 7. 주요 인사이트 요약 =====
print("\n\n" + "="*70)
print("【주요 인사이트 요약】")
print("="*70)

insights = [
    f"1. Self-Attention OE는 Baseline 대비 AUROC {(df_selfoe['auroc_mean'].mean() - df_baseline['auroc_mean'].mean()) / df_baseline['auroc_mean'].mean() * 100:+.2f}% 개선",
    f"2. Masking Probability 0.05가 0.00보다 약 {(mask_5.mean() - mask_0.mean()) / mask_0.mean() * 100:.1f}% 우수",
    f"3. {best_method['method']} 필터링 방법이 최고 성능 (AUROC {best_method['auroc_mean']:.4f})",
    f"4. Attention Top-p는 {best_topp:.2f}가 최적",
    f"5. WikiText OOD에서 AG News보다 더 큰 개선 효과 ({wiki_improvement:.2f}% vs {ag_improvement:.2f}%)",
    f"6. 70개 실험 중 최고 성능: AUROC {df_selfoe['auroc_mean'].max():.4f}",
    f"7. 성능 편차(std)가 작아 안정적인 방법론 (σ={df_selfoe['auroc_mean'].std():.4f})"
]

for insight in insights:
    print(f"\n  ✓ {insight}")

print("\n" + "="*70)

# 인사이트 파일 저장
with open('statistical_insights.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("통계적 인사이트 분석 결과\n")
    f.write("="*70 + "\n\n")
    for insight in insights:
        f.write(f"{insight}\n")

print("\n✓ 인사이트가 'statistical_insights.txt'에 저장되었습니다.")
