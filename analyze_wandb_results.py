#!/usr/bin/env python3
"""
WANDB 실험 결과 분석 스크립트 (수정 버전)
98개의 Self-Attention OE sweep 실험 결과를 분석하고 시각화
"""

import wandb
import pandas as pd
import numpy as np
import json

# WANDB 설정
WANDB_API_KEY = "e7e799fb8b3cb64fd75e6579b4829448ed0b9103"
WANDB_PROJECT = "20251009-NEWSGROUP-4090"
wandb.login(key=WANDB_API_KEY)

# API 초기화
api = wandb.Api()

# 프로젝트에서 모든 runs 가져오기
runs = api.runs(f"bang001-ku/{WANDB_PROJECT}")

print(f"총 {len(runs)}개의 실험 발견\n")

# 데이터 수집
data = []
for i, run in enumerate(runs):
    if i % 10 == 0:
        print(f"처리 중: {i+1}/{len(runs)}")

    # 실험 설정 및 결과 추출
    config = run.config
    summary = run.summary._json_dict

    row = {
        'run_id': run.id,
        'run_name': run.name,
        'state': run.state,
        # Config
        'use_self_attention_oe': config.get('use_self_attention_oe', False),
        'attention_top_p': config.get('attention_top_p'),
        'masking_probability': config.get('masking_probability'),
        'attention_filtering_method': config.get('attention_filtering_method', 'N/A'),
        'oe_source': config.get('oe_source', 'Unknown'),
        'oe_method': config.get('oe_method', 'Unknown'),
        'dataset': config.get('dataset'),
        'num_epochs': config.get('num_epochs'),
        # OOD 성능 메트릭
        'auroc_ag_news': summary.get('ood/ag_news/AUROC'),
        'aupr_ag_news': summary.get('ood/ag_news/AUPR'),
        'fpr95_ag_news': summary.get('ood/ag_news/FPR95'),
        'auroc_wikitext': summary.get('ood/wikitext/AUROC'),
        'aupr_wikitext': summary.get('ood/wikitext/AUPR'),
        'fpr95_wikitext': summary.get('ood/wikitext/FPR95'),
        # 평균 메트릭
        'auroc_mean': summary.get('summary/AUROC_mean'),
        'aupr_mean': summary.get('summary/AUPR_mean'),
        'fpr95_mean': summary.get('summary/FPR95_mean'),
        # In-distribution 메트릭
        'id_val_f1': summary.get('id/val_f1_macro'),
        'val_accuracy': summary.get('val_accuracy'),
        'val_auroc': summary.get('val_auroc'),
        # 카운트 정보
        'generated_oe': summary.get('counts/GeneratedOE', 0),
        'processed_oe': summary.get('counts/ProcessedOE', 0),
        # 기타
        'epoch': summary.get('epoch'),
        'runtime': summary.get('_runtime'),
    }
    data.append(row)

# DataFrame 생성
df = pd.DataFrame(data)

print(f"\n총 수집된 실험: {len(df)}개")
print(f"Self-Attention OE 실험: {df['use_self_attention_oe'].sum()}개")
print(f"Standard 실험: {(~df['use_self_attention_oe']).sum()}개")

# 완료된 실험만 필터링
df_completed = df[df['state'] == 'finished'].copy()
print(f"완료된 실험: {len(df_completed)}개")

# Self-Attention OE 실험만 필터링
df_self_oe = df_completed[df_completed['use_self_attention_oe'] == True].copy()
print(f"완료된 Self-Attention OE 실험: {len(df_self_oe)}개")

# 데이터 저장
df.to_csv('wandb_all_results.csv', index=False)
df_completed.to_csv('wandb_completed_results.csv', index=False)
df_self_oe.to_csv('wandb_self_oe_results.csv', index=False)

print(f"\n결과 저장 완료:")
print(f"  - wandb_all_results.csv: 전체 {len(df)}개")
print(f"  - wandb_completed_results.csv: 완료된 {len(df_completed)}개")
print(f"  - wandb_self_oe_results.csv: Self-OE {len(df_self_oe)}개")

# 결측치 확인
print(f"\n=== 주요 메트릭 결측치 확인 (Self-OE) ===")
missing_cols = ['auroc_mean', 'aupr_mean', 'fpr95_mean', 'attention_top_p', 'masking_probability']
for col in missing_cols:
    missing = df_self_oe[col].isnull().sum()
    print(f"  {col}: {missing}/{len(df_self_oe)} 결측")

# 기본 통계
if len(df_self_oe) > 0:
    print("\n=== Self-Attention OE 기본 통계 ===")
    stat_cols = ['auroc_mean', 'aupr_mean', 'fpr95_mean', 'auroc_ag_news', 'auroc_wikitext']
    print(df_self_oe[stat_cols].describe().round(4))

    # 파라미터별 분석
    print("\n=== 1. attention_top_p별 평균 성능 ===")
    if df_self_oe['attention_top_p'].notna().sum() > 0:
        top_p_stats = df_self_oe.groupby('attention_top_p').agg({
            'auroc_mean': ['mean', 'std', 'count'],
            'aupr_mean': ['mean', 'std'],
            'fpr95_mean': ['mean', 'std'],
            'auroc_ag_news': ['mean', 'std'],
            'auroc_wikitext': ['mean', 'std']
        }).round(4)
        print(top_p_stats)
    else:
        print("  데이터 없음")

    print("\n=== 2. masking_probability별 평균 성능 ===")
    if df_self_oe['masking_probability'].notna().sum() > 0:
        mask_stats = df_self_oe.groupby('masking_probability').agg({
            'auroc_mean': ['mean', 'std', 'count'],
            'aupr_mean': ['mean', 'std'],
            'fpr95_mean': ['mean', 'std'],
            'auroc_ag_news': ['mean', 'std'],
            'auroc_wikitext': ['mean', 'std']
        }).round(4)
        print(mask_stats)
    else:
        print("  데이터 없음")

    print("\n=== 3. attention_filtering_method별 평균 성능 ===")
    filter_stats = df_self_oe.groupby('attention_filtering_method').agg({
        'auroc_mean': ['mean', 'std', 'count'],
        'aupr_mean': ['mean', 'std'],
        'fpr95_mean': ['mean', 'std']
    }).round(4)
    print(filter_stats)

    # 최고 성능 조합
    print("\n=== 최고 성능 실험 (AUROC 기준, Top 10) ===")
    best_runs = df_self_oe.nlargest(10, 'auroc_mean')[
        ['run_name', 'attention_top_p', 'masking_probability', 'attention_filtering_method',
         'auroc_mean', 'aupr_mean', 'fpr95_mean', 'auroc_ag_news', 'auroc_wikitext']
    ]
    print(best_runs.to_string())

    # 파라미터 조합별 성능 (결측치 제외)
    print("\n=== 파라미터 조합별 평균 성능 ===")
    df_with_params = df_self_oe[
        df_self_oe['attention_top_p'].notna() &
        df_self_oe['masking_probability'].notna()
    ].copy()

    if len(df_with_params) > 0:
        combo_stats = df_with_params.groupby(['attention_top_p', 'masking_probability']).agg({
            'auroc_mean': ['mean', 'std', 'count'],
            'aupr_mean': ['mean', 'std'],
            'fpr95_mean': ['mean', 'std']
        }).round(4)
        print(combo_stats)
    else:
        print("  파라미터 조합 데이터 없음")

    # Baseline 비교
    print("\n=== Baseline (Standard) vs Self-Attention OE 비교 ===")
    df_baseline = df_completed[df_completed['use_self_attention_oe'] == False]

    if len(df_baseline) > 0:
        print("Baseline (Standard):")
        print(f"  AUROC mean: {df_baseline['auroc_mean'].mean():.4f} ± {df_baseline['auroc_mean'].std():.4f}")
        print(f"  AUPR mean: {df_baseline['aupr_mean'].mean():.4f} ± {df_baseline['aupr_mean'].std():.4f}")
        print(f"  FPR95 mean: {df_baseline['fpr95_mean'].mean():.4f} ± {df_baseline['fpr95_mean'].std():.4f}")

    print("\nSelf-Attention OE:")
    print(f"  AUROC mean: {df_self_oe['auroc_mean'].mean():.4f} ± {df_self_oe['auroc_mean'].std():.4f}")
    print(f"  AUPR mean: {df_self_oe['aupr_mean'].mean():.4f} ± {df_self_oe['aupr_mean'].std():.4f}")
    print(f"  FPR95 mean: {df_self_oe['fpr95_mean'].mean():.4f} ± {df_self_oe['fpr95_mean'].std():.4f}")

    if len(df_baseline) > 0:
        auroc_improvement = ((df_self_oe['auroc_mean'].mean() - df_baseline['auroc_mean'].mean()) /
                            df_baseline['auroc_mean'].mean() * 100)
        print(f"\n개선률:")
        print(f"  AUROC: {auroc_improvement:+.2f}%")

else:
    print("\nSelf-Attention OE 실험 데이터가 없습니다.")

print("\n분석 완료!")
