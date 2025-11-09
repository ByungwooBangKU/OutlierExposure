#!/usr/bin/env python3
"""WANDB 실험의 실제 config 및 메트릭 확인"""

import wandb
import json

WANDB_API_KEY = "e7e799fb8b3cb64fd75e6579b4829448ed0b9103"
WANDB_PROJECT = "20251009-NEWSGROUP-4090"
wandb.login(key=WANDB_API_KEY)

api = wandb.Api()
runs = api.runs(f"bang001-ku/{WANDB_PROJECT}")

print(f"총 {len(runs)}개의 실험 발견\n")

# 첫 3개 실험의 구조 확인
for i, run in enumerate(list(runs)[:3]):
    print(f"=== Run {i+1}: {run.name} ===")
    print(f"Run ID: {run.id}")
    print(f"\nConfig 키:")
    for key in sorted(run.config.keys()):
        print(f"  {key}: {run.config.get(key)}")

    print(f"\nSummary 키 (상위 20개):")
    summary_keys = sorted(run.summary._json_dict.keys())[:20]
    for key in summary_keys:
        value = run.summary._json_dict.get(key)
        if not key.startswith('_'):
            print(f"  {key}: {value}")

    print("\n" + "="*60 + "\n")

# 모든 config 키 수집
all_config_keys = set()
all_summary_keys = set()

for run in runs:
    all_config_keys.update(run.config.keys())
    all_summary_keys.update(run.summary._json_dict.keys())

print("=== 전체 Config 키 ===")
for key in sorted(all_config_keys):
    print(f"  - {key}")

print("\n=== 전체 Summary 키 (메트릭) ===")
for key in sorted(all_summary_keys):
    if not key.startswith('_'):
        print(f"  - {key}")
