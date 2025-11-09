#!/usr/bin/env python3
"""WANDB 프로젝트 목록 확인"""

import wandb

WANDB_API_KEY = "e7e799fb8b3cb64fd75e6579b4829448ed0b9103"
wandb.login(key=WANDB_API_KEY)

api = wandb.Api()

# 현재 사용자 정보 확인
print(f"현재 로그인한 사용자: {api.viewer}")

# 프로젝트 목록 확인
try:
    # 여러 가능한 entity 시도
    entities = ['bang001-ku', 'bang001', 'byungwoobang', 'ByungwooBangKU']

    for entity in entities:
        print(f"\n=== Entity: {entity} ===")
        try:
            projects = api.projects(entity=entity)
            print(f"프로젝트 목록:")
            for project in projects:
                print(f"  - {project.name} ({project.entity}/{project.name})")
        except Exception as e:
            print(f"  오류: {e}")

    # 특정 프로젝트 이름으로 검색
    print("\n=== 프로젝트 검색 시도 ===")
    possible_names = [
        "20251009-NEWSGROUP-4090",
        "NEWSGROUP-4090",
        "newsgroup",
        "outlier-exposure",
        "oe"
    ]

    for entity in ['bang001-ku', 'bang001']:
        for proj_name in possible_names:
            try:
                runs = api.runs(f"{entity}/{proj_name}", per_page=1)
                count = len(list(runs))
                if count > 0:
                    print(f"✓ 발견: {entity}/{proj_name} ({count} runs)")
            except:
                pass

except Exception as e:
    print(f"전체 오류: {e}")
