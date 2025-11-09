#!/usr/bin/env python3
"""
추가 실험 파라미터 제안 분석

현재 실험 결과를 기반으로 다음 단계 실험을 위한 파라미터 범위 제안
"""

import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('wandb_self_oe_results.csv')

print("="*80)
print("현재 실험 파라미터 분석 및 추가 실험 제안")
print("="*80)

# 1. 현재 실험된 파라미터 확인
print("\n【현재 실험된 파라미터 범위】")
print("-"*80)
print(f"attention_top_p: {sorted(df['attention_top_p'].unique())}")
print(f"masking_probability: {sorted(df['masking_probability'].unique())}")
print(f"attention_filtering_method: {sorted(df['attention_filtering_method'].unique())}")

# 2. 성능 분석
print("\n【성능 분석 요약】")
print("-"*80)
print(f"최고 AUROC: {df['auroc_mean'].max():.4f}")
print(f"평균 AUROC: {df['auroc_mean'].mean():.4f} ± {df['auroc_mean'].std():.4f}")

# Masking 효과
mask_0 = df[df['masking_probability'] == 0.00]['auroc_mean'].mean()
mask_5 = df[df['masking_probability'] == 0.05]['auroc_mean'].mean()
print(f"\nMasking 0.00: {mask_0:.4f}")
print(f"Masking 0.05: {mask_5:.4f} (개선: +{(mask_5-mask_0)/mask_0*100:.2f}%)")

# Top-p 효과
top_p_stats = df.groupby('attention_top_p')['auroc_mean'].agg(['mean', 'std'])
print(f"\nTop-p별 성능:")
for idx, row in top_p_stats.iterrows():
    print(f"  {idx:.2f}: {row['mean']:.4f} ± {row['std']:.4f}")

# Filtering method 효과
filter_stats = df.groupby('attention_filtering_method')['auroc_mean'].agg(['mean', 'std', 'count'])
filter_stats = filter_stats.sort_values('mean', ascending=False)
print(f"\nFiltering Method별 성능 (상위 3개):")
for idx, row in filter_stats.head(3).iterrows():
    print(f"  {idx}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")

# 3. 추가 실험 제안
print("\n\n" + "="*80)
print("【추가 실험 파라미터 제안】")
print("="*80)

recommendations = []

# 제안 1: Masking Probability 세밀화
print("\n1️⃣  Masking Probability 세밀화 (Fine-tuning around optimal)")
print("-"*80)
print("현재: 0.00, 0.05 → 0.05가 +4.15% 우수")
print("제안: 0.05 주변 세밀 탐색으로 최적값 찾기")
print()
print("  --masking_probabilities 0.03,0.04,0.05,0.06,0.07,0.08")
print()
print("근거:")
print("  • Masking 0.05가 통계적으로 매우 유의미하게 우수 (p<0.001)")
print("  • 0.05 주변 값으로 더 좋은 성능 가능성")
print("  • 0.03~0.08 범위로 최적 sweet spot 탐색")

recommendations.append({
    'name': 'Masking Fine-tuning',
    'param': '--masking_probabilities',
    'values': '0.03,0.04,0.05,0.06,0.07,0.08',
    'priority': 'HIGH',
    'expected_gain': '0.5-1.0% AUROC 추가 개선 가능'
})

# 제안 2: Attention Top-p 확장
print("\n\n2️⃣  Attention Top-p 확장 (Broader exploration)")
print("-"*80)
print("현재: 0.15, 0.25, 0.35 → 0.25가 최적 (차이 작음)")
print("제안: 더 넓은 범위 + 최적값 주변 세밀화")
print()
print("  옵션 A (넓은 범위): --attention_top_p_values 0.10,0.20,0.30,0.40,0.50")
print("  옵션 B (세밀화):   --attention_top_p_values 0.20,0.22,0.24,0.26,0.28,0.30")
print()
print("근거:")
print("  • 현재 top-p 간 차이가 통계적으로 유의미하지 않음 (p=0.362)")
print("  • 더 넓은 범위 탐색으로 비선형 효과 확인 필요")
print("  • 또는 0.25 주변 세밀 탐색으로 미세 최적화")

recommendations.append({
    'name': 'Top-p Exploration',
    'param': '--attention_top_p_values',
    'values': '0.10,0.20,0.30,0.40,0.50 (또는 0.20,0.22,0.24,0.26,0.28,0.30)',
    'priority': 'MEDIUM',
    'expected_gain': '현재 robust하므로 큰 개선 어려움, 안정성 확인 목적'
})

# 제안 3: Attention Stages 변경
print("\n\n3️⃣  Attention Stages 확장 (Stage exploration)")
print("-"*80)
print("현재: stage2만 실험")
print("제안: stage3, both 추가 실험")
print()
print("  --attention_stages stage2,stage3,both")
print()
print("근거:")
print("  • Stage2는 attention cache 생성 단계")
print("  • Stage3는 실제 OE 학습 단계 - 성능 차이 가능성")
print("  • Both는 stage2+stage3 동시 수행")
print("  • 각 stage별 최적 파라미터가 다를 수 있음")

recommendations.append({
    'name': 'Stage Exploration',
    'param': '--attention_stages',
    'values': 'stage2,stage3,both',
    'priority': 'HIGH',
    'expected_gain': 'Stage별 특성에 따라 1-2% 개선 가능'
})

# 제안 4: Loss Weights 조정
print("\n\n4️⃣  Loss Weights 조정 (Loss balancing)")
print("-"*80)
print("현재: oe_uniform_loss_weight=1.0, self_attention_loss_weight=1.0 (고정)")
print("제안: Loss weight 비율 조정으로 균형 탐색")
print()
print("  --oe_uniform_loss_weights 0.5,1.0,1.5,2.0")
print("  --self_attention_loss_weights 0.5,1.0,1.5,2.0")
print()
print("근거:")
print("  • ID classification과 OE loss 간 균형 중요")
print("  • Hendrycks는 λ=1.0 사용했지만, Self-OE는 다를 수 있음")
print("  • 0.5~2.0 범위로 최적 비율 탐색")

recommendations.append({
    'name': 'Loss Weight Tuning',
    'param': '--oe_uniform_loss_weights, --self_attention_loss_weights',
    'values': '0.5,1.0,1.5,2.0 (각각)',
    'priority': 'MEDIUM',
    'expected_gain': '0.5-1.5% AUROC 개선 가능'
})

# 제안 5: Attention Top-k 실험
print("\n\n5️⃣  Attention Top-k 값 도입 (Top-k filtering)")
print("-"*80)
print("현재: top-k=None (사용 안 함)")
print("제안: Top-k 값 도입으로 토큰 선택 강화")
print()
print("  --attention_top_k_values 3,5,10,20,50")
print()
print("근거:")
print("  • Top-p만으로는 너무 많은/적은 토큰 선택 가능")
print("  • Top-k로 선택 토큰 수를 명시적으로 제한")
print("  • 작은 k (3-10): 핵심 토큰만, 큰 k (20-50): 넓은 범위")

recommendations.append({
    'name': 'Top-k Introduction',
    'param': '--attention_top_k_values',
    'values': '3,5,10,20,50',
    'priority': 'MEDIUM-HIGH',
    'expected_gain': '1-2% AUROC 개선 가능 (핵심 토큰 타겟팅)'
})

# 제안 6: 최고 성능 조합 Ablation Study
print("\n\n6️⃣  최고 성능 조합 주변 Ablation Study")
print("-"*80)
print(f"현재 최고 성능: AUROC 0.8264")
print("  - top-p=0.25, masking=0.05, top_k_avg_elbow_lower")
print()
print("제안: 최고 성능 조합 주변 집중 탐색")
print()
print("  --attention_top_p_values 0.23,0.25,0.27")
print("  --masking_probabilities 0.04,0.05,0.06")
print("  --attention_filtering_method만 top_k_avg_elbow_lower로 고정")
print()
print("근거:")
print("  • 최고 성능 조합 주변에 더 좋은 점이 있을 가능성")
print("  • 계산 비용 절감 (한 가지 filtering만)")
print("  • 미세 최적화로 0.83+ AUROC 달성 가능성")

recommendations.append({
    'name': 'Best Config Ablation',
    'param': '--attention_top_p_values, --masking_probabilities',
    'values': '0.23,0.25,0.27 / 0.04,0.05,0.06',
    'priority': 'HIGH',
    'expected_gain': '0.5-1.0% AUROC 추가 개선 → 0.83+ 달성 가능'
})

# 제안 7: Epoch 수 증가
print("\n\n7️⃣  Training Epochs 증가 (Longer training)")
print("-"*80)
print("현재: num_epochs=5")
print("제안: 학습 에폭 증가로 수렴 확인")
print()
print("  --num_epochs 7 (또는 10)")
print()
print("근거:")
print("  • 5 epoch에서 underfitting 가능성")
print("  • 더 긴 학습으로 성능 향상 여부 확인")
print("  • Early stopping 있으므로 overfitting 위험 낮음")

recommendations.append({
    'name': 'Longer Training',
    'param': '--num_epochs',
    'values': '7 or 10',
    'priority': 'LOW-MEDIUM',
    'expected_gain': '0.3-0.7% AUROC 개선 가능 (수렴 여부에 따라)'
})

# 제안 요약 테이블
print("\n\n" + "="*80)
print("【추가 실험 우선순위 요약】")
print("="*80)

print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ 우선순위 │ 실험명                     │ 기대 효과                          │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print("│ HIGH     │ 1. Masking Fine-tuning     │ 0.5-1.0% AUROC 추가 개선          │")
print("│ HIGH     │ 3. Stage Exploration       │ 1-2% AUROC 개선 (stage별 특성)    │")
print("│ HIGH     │ 6. Best Config Ablation    │ 0.5-1.0% 개선 → 0.83+ 달성 가능   │")
print("│ MED-HIGH │ 5. Top-k Introduction      │ 1-2% AUROC 개선 (핵심 토큰)       │")
print("│ MEDIUM   │ 2. Top-p Exploration       │ 안정성 확인 (큰 개선 어려움)      │")
print("│ MEDIUM   │ 4. Loss Weight Tuning      │ 0.5-1.5% AUROC 개선               │")
print("│ LOW-MED  │ 7. Longer Training         │ 0.3-0.7% AUROC 개선               │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

# 구체적 명령어 예시
print("\n\n" + "="*80)
print("【추천 실험 명령어】")
print("="*80)

print("\n✅ 실험 세트 1: 최우선 실험 (Best Config Ablation + Masking Fine-tuning)")
print("-"*80)
cmd1 = """python scripts/run_oe_sweep.py \\
  --dataset 20newsgroups \\
  --mode self_attention_oe \\
  --attention_generation_modes staged \\
  --attention_stages stage2 \\
  --attention_top_p_values 0.23,0.25,0.27 \\
  --masking_probabilities 0.03,0.04,0.05,0.06,0.07 \\
  --num_epochs 5 \\
  --attention_cache_base simplified_oe_experiments/oe_cache \\
  --output_dir sweeps/oe/ablation_best \\
  --extra_args "--attention_filtering_method top_k_avg_elbow_lower"
"""
print(cmd1)
print(f"예상 실험 수: 3 (top-p) × 5 (masking) × 1 (filter) = 15개")
print(f"예상 소요 시간: ~5-6시간 (실험당 20분 기준)")

print("\n✅ 실험 세트 2: Stage 확장 실험")
print("-"*80)
cmd2 = """python scripts/run_oe_sweep.py \\
  --dataset 20newsgroups \\
  --mode self_attention_oe \\
  --attention_generation_modes staged \\
  --attention_stages stage2,stage3,both \\
  --attention_top_p_values 0.25 \\
  --masking_probabilities 0.05 \\
  --num_epochs 5 \\
  --attention_cache_base simplified_oe_experiments/oe_cache \\
  --output_dir sweeps/oe/stage_exploration
"""
print(cmd2)
print(f"예상 실험 수: 3 (stages) × 5 (filters) = 15개")
print(f"예상 소요 시간: ~5-6시간")

print("\n✅ 실험 세트 3: Top-k 도입 실험")
print("-"*80)
cmd3 = """python scripts/run_oe_sweep.py \\
  --dataset 20newsgroups \\
  --mode self_attention_oe \\
  --attention_generation_modes staged \\
  --attention_stages stage2 \\
  --attention_top_p_values 0.25 \\
  --masking_probabilities 0.05 \\
  --attention_top_k_values 3,5,10,20,50 \\
  --num_epochs 5 \\
  --attention_cache_base simplified_oe_experiments/oe_cache \\
  --output_dir sweeps/oe/topk_exploration
"""
print(cmd3)
print(f"예상 실험 수: 5 (top-k) × 5 (filters) = 25개")
print(f"예상 소요 시간: ~8-10시간")

print("\n✅ 실험 세트 4: Loss Weight 조정")
print("-"*80)
cmd4 = """python scripts/run_oe_sweep.py \\
  --dataset 20newsgroups \\
  --mode self_attention_oe \\
  --attention_generation_modes staged \\
  --attention_stages stage2 \\
  --attention_top_p_values 0.25 \\
  --masking_probabilities 0.05 \\
  --oe_uniform_loss_weights 0.5,1.0,1.5,2.0 \\
  --self_attention_loss_weights 0.5,1.0,1.5,2.0 \\
  --num_epochs 5 \\
  --attention_cache_base simplified_oe_experiments/oe_cache \\
  --output_dir sweeps/oe/loss_tuning \\
  --extra_args "--attention_filtering_method top_k_avg_elbow_lower"
"""
print(cmd4)
print(f"예상 실험 수: 4 (oe_weight) × 4 (sa_weight) = 16개")
print(f"예상 소요 시간: ~5-6시간")

# 최종 추천
print("\n\n" + "="*80)
print("【최종 추천】")
print("="*80)
print("""
1️⃣  즉시 실행 (우선순위 HIGH):
   • 실험 세트 1: Best Config Ablation (15개 실험, ~6시간)
     → 0.83+ AUROC 달성 가능성이 가장 높음

2️⃣  병렬 실행 가능 시:
   • 실험 세트 1 + 실험 세트 2 (Stage Exploration) 동시 실행
     → 총 30개 실험, 각각 독립적이므로 병렬 가능

3️⃣  리소스 여유 시 추가:
   • 실험 세트 3 (Top-k) 또는 세트 4 (Loss Weight)
     → 더 넓은 파라미터 공간 탐색

4️⃣  장기 실험:
   • num_epochs=7 또는 10으로 모든 실험 재실행
     → 수렴 여부 확인 및 최종 성능 최적화

💡 계산 비용 대비 효과가 가장 높은 조합:
   실험 세트 1 (Best Config Ablation) → 15개 실험으로 최대 효과
""")

print("\n" + "="*80)

# 저장
with open('parameter_recommendations.txt', 'w', encoding='utf-8') as f:
    f.write("추가 실험 파라미터 제안\n")
    f.write("="*80 + "\n\n")
    for rec in recommendations:
        f.write(f"{rec['name']}\n")
        f.write(f"  우선순위: {rec['priority']}\n")
        f.write(f"  파라미터: {rec['param']}\n")
        f.write(f"  값: {rec['values']}\n")
        f.write(f"  기대 효과: {rec['expected_gain']}\n\n")

print("\n✓ 제안사항이 'parameter_recommendations.txt'에 저장되었습니다.")
