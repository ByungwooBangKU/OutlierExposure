#!/usr/bin/env python3
"""Hyperparameter sweep runner for oe.py covering baseline/traditional/self-attention/both OE modes."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

# Recommended staged self-attention OE ranges tuned for ID-derived OE exploration.
DEFAULT_ATTENTION_TOP_PS = [0.20, 0.25, 0.30]
DEFAULT_MASKING_PROBABILITIES = [0.05]
DEFAULT_OE_UNIFORM_LOSS_WEIGHTS = [1.0]  # Hendrycks uses λ=1.0
DEFAULT_SELF_ATTENTION_LOSS_WEIGHTS = [1.0]  # Deprecated, kept for compatibility
DEFAULT_ATTENTION_TOP_KS: Sequence[int | None] = (None,)
DEFAULT_ATTENTION_GENERATION_MODES = ["staged"]
DEFAULT_ATTENTION_STAGES = ["stage3"]
# DEFAULT_STAGE3_MASKING_OPTIONS removed - masking is always enabled (Hendrycks requirement)


@dataclass
class OESweepConfig:
    dataset: str
    mode: str
    model_name: str | None
    attention_generation_modes: Sequence[str]
    attention_stages: Sequence[str]
    attention_cache_base: Path | None
    attention_stage2_shard_size: int | None
    # stage3_masking_options removed - masking is always enabled
    attention_top_p_values: Sequence[float]
    masking_probabilities: Sequence[float]
    oe_uniform_loss_weights: Sequence[float]
    self_attention_loss_weights: Sequence[float]
    attention_top_k_values: Sequence[int | None]
    oe_max_samples: int | None
    num_epochs: int | None
    batch_size: int | None
    output_dir: Path
    extra_args: Sequence[str]
    transformers_offline: bool = True
    overwrite: bool = False
    disable_model_checkpoint: bool = True
    precompute_stage2: bool = True
    stage2_overwrite: bool = False
    stage3_use_cached_standard_model: bool = False  # Changed: Use fresh initialization (Hendrycks-style)
    stage3_standard_checkpoint: Path | None = None


def _directory_has_content(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False
    except OSError:
        return False


def prepare_huggingface_cache(env: dict[str, str]) -> Path:
    hf_home = Path(env.get("HF_HOME", Path.cwd() / "huggingface_cache")).expanduser().resolve()
    env.setdefault("HF_HOME", str(hf_home))

    datasets_cache = Path(env.get("HF_DATASETS_CACHE", hf_home / "datasets")).expanduser().resolve()
    transformers_cache = Path(env.get("TRANSFORMERS_CACHE", hf_home / "transformers")).expanduser().resolve()

    datasets_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)

    env["HF_DATASETS_CACHE"] = str(datasets_cache)
    env["TRANSFORMERS_CACHE"] = str(transformers_cache)

    return hf_home


def huggingface_cache_ready(env: dict[str, str]) -> bool:
    datasets_cache = Path(env.get("HF_DATASETS_CACHE", "")).expanduser()
    transformers_cache = Path(env.get("TRANSFORMERS_CACHE", "")).expanduser()
    return _directory_has_content(datasets_cache) and _directory_has_content(transformers_cache)


def parse_float_list(values: str) -> Sequence[float]:
    return tuple(float(item) for item in values.split(',') if item)


def parse_optional_int_list(values: str) -> Sequence[int | None]:
    tokens = [token.strip() for token in values.split(',') if token]
    result: list[int | None] = []
    for token in tokens:
        if token.lower() in {"none", "null"}:
            result.append(None)
        else:
            result.append(int(token))
    return tuple(result)


def parse_bool_options(values: str) -> Sequence[bool]:
    mapping = {
        'on': True,
        'off': False,
        'true': True,
        'false': False,
        '1': True,
        '0': False,
    }
    tokens = [token.strip().lower() for token in values.split(',') if token]
    parsed: list[bool] = []
    for token in tokens:
        if token not in mapping:
            raise ValueError(f"Invalid boolean option '{token}'. Use on/off/true/false/1/0")
        parsed.append(mapping[token])
    return tuple(parsed)


def parse_str_list(values: str) -> Sequence[str]:
    return tuple(token.strip() for token in values.split(',') if token.strip())


def parse_args() -> OESweepConfig:
    parser = argparse.ArgumentParser(description="Run hyper-parameter sweeps for oe.py")
    parser.add_argument("--dataset", default="20newsgroups")
    parser.add_argument("--mode", default="both_oe",
                        choices=["baseline", "traditional_oe", "self_attention_oe", "both_oe"])
    parser.add_argument("--model_name", default="",
                        help="Optional model override passed through to oe.py (e.g., gru-baseline)")
    parser.add_argument("--attention_generation_modes", default=",".join(DEFAULT_ATTENTION_GENERATION_MODES),
                        help="Comma separated attention generation modes (on_the_fly, staged)")
    parser.add_argument("--attention_stages", default=",".join(DEFAULT_ATTENTION_STAGES),
                        help="Comma separated staged attention phases (stage2, stage3, both)")
    parser.add_argument("--attention_cache_base", default="",
                        help="Base directory for staged attention cache (reused across runs)")
    parser.add_argument("--attention_stage2_shard_size", type=int, default=None,
                        help="Override Stage2 shard size when caching attention scores")
    # --stage3_masking_options removed: masking is always enabled (Hendrycks requirement)
    parser.add_argument("--stage3_standard_checkpoint", default="",
                        help="Optional path for saving/reusing the Standard checkpoint in staged stage3 sweeps")
    parser.add_argument("--no_stage3_cached_model", action="store_true",
                        help="Disable reuse of cached Standard model when running staged stage3 sweeps")
    parser.add_argument("--attention_top_p_values", default=",".join(str(v) for v in DEFAULT_ATTENTION_TOP_PS))
    parser.add_argument("--masking_probabilities", default=",".join(str(v) for v in DEFAULT_MASKING_PROBABILITIES))
    parser.add_argument("--oe_uniform_loss_weights", default=",".join(str(v) for v in DEFAULT_OE_UNIFORM_LOSS_WEIGHTS))
    parser.add_argument("--self_attention_loss_weights", default=",".join(str(v) for v in DEFAULT_SELF_ATTENTION_LOSS_WEIGHTS))
    parser.add_argument("--attention_top_k_values", default=",".join("none" if v is None else str(v) for v in DEFAULT_ATTENTION_TOP_KS))
    parser.add_argument("--oe_max_samples", type=int, default=None,
                        help="Cap for OE sampling (None keeps dataset size)")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", default="sweeps/oe")
    parser.add_argument("--extra_args", default="")
    parser.add_argument("--no_offline", action="store_true",
                        help="Disable TRANSFORMERS_OFFLINE=1 during runs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_model_checkpoints", action="store_true",
                        help="Skip passing --disable_model_checkpoint to oe.py")
    parser.add_argument("--no_stage2_warmup", action="store_true",
                        help="Skip precomputing Stage2 cache in staged mode")
    parser.add_argument("--stage2_overwrite", action="store_true",
                        help="Re-run Stage2 warmup even if log exists")
    args = parser.parse_args()

    # stage3_masking_options removed - masking is always enabled
    # Use fresh initialization by default (Hendrycks-style), unless explicitly disabled
    if args.no_stage3_cached_model:
        stage3_use_cached = False
    else:
        # Changed default: Fresh initialization (Hendrycks-style)
        stage3_use_cached = False
    stage3_checkpoint = Path(args.stage3_standard_checkpoint).resolve() if args.stage3_standard_checkpoint else None

    return OESweepConfig(
        dataset=args.dataset,
        mode=args.mode,
        model_name=args.model_name or None,
        attention_generation_modes=parse_str_list(args.attention_generation_modes),
        attention_stages=parse_str_list(args.attention_stages),
        attention_cache_base=Path(args.attention_cache_base).resolve() if args.attention_cache_base else None,
        attention_stage2_shard_size=args.attention_stage2_shard_size,
        # stage3_masking_options removed
        attention_top_p_values=parse_float_list(args.attention_top_p_values),
        masking_probabilities=parse_float_list(args.masking_probabilities),
        oe_uniform_loss_weights=parse_float_list(args.oe_uniform_loss_weights) or tuple(DEFAULT_OE_UNIFORM_LOSS_WEIGHTS),
        self_attention_loss_weights=parse_float_list(args.self_attention_loss_weights),
        attention_top_k_values=parse_optional_int_list(args.attention_top_k_values),
        oe_max_samples=args.oe_max_samples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
        extra_args=tuple(args.extra_args.split()) if args.extra_args else (),
        transformers_offline=not args.no_offline,
        overwrite=args.overwrite,
        disable_model_checkpoint=not args.keep_model_checkpoints,
        precompute_stage2=not args.no_stage2_warmup,
        stage2_overwrite=args.stage2_overwrite,
        stage3_use_cached_standard_model=stage3_use_cached,
        stage3_standard_checkpoint=stage3_checkpoint,
    )


def run_command(cmd: Sequence[str], log_path: Path, env: dict[str, str], *, stream_to_console: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# Command: {' '.join(cmd)}\n")
        log_file.write(f"# Started: {datetime.now().isoformat()}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None  # for type checker
        try:
            for line in process.stdout:
                log_file.write(line)
                if stream_to_console:
                    print(line.rstrip())
        finally:
            process.stdout.close()

        returncode = process.wait()
        log_file.write(f"\n# Finished: {datetime.now().isoformat()} (returncode={returncode})\n")
    return returncode


def read_results(dataset: str) -> pd.DataFrame:
    results_path = Path("simplified_oe_experiments") / "results" / f"osr_results_{dataset}.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Expected results CSV not found: {results_path}")
    return pd.read_csv(results_path)


def append_summary(summary_path: Path, df: pd.DataFrame) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(summary_path, index=False)


def build_base_command(config: OESweepConfig, *, override_mode: str | None = None) -> List[str]:
    cmd = [
        "python", "oe.py",
        "--dataset", config.dataset,
        "--mode", override_mode or config.mode,
    ]
    if config.model_name:
        cmd.extend(["--model_name", config.model_name])
    if config.disable_model_checkpoint:
        cmd.append("--disable_model_checkpoint")
    if config.attention_stage2_shard_size is not None and config.attention_stage2_shard_size > 0:
        cmd.extend(["--attention_stage2_shard_size", str(config.attention_stage2_shard_size)])
    if config.oe_max_samples is not None:
        cmd.extend(["--oe_max_samples", str(config.oe_max_samples)])
    if config.num_epochs is not None:
        cmd.extend(["--num_epochs", str(config.num_epochs)])
    if config.batch_size is not None:
        cmd.extend(["--batch_size", str(config.batch_size)])
    if config.extra_args:
        cmd.extend(config.extra_args)
    return cmd


def maybe_run_stage2(config: OESweepConfig, env: dict[str, str]) -> None:
    if "staged" not in config.attention_generation_modes or not config.precompute_stage2:
        return

    if config.attention_cache_base is None:
        print("⚠️ Stage2 warmup skipped because --attention_cache_base is not set.")
        return

    stage2_log = config.output_dir / "stage2_warmup.log"
    if stage2_log.exists() and not config.stage2_overwrite:
        print("Stage2 cache already present (log exists). Skipping warmup.")
        return

    config.attention_cache_base.mkdir(parents=True, exist_ok=True)

    top_p = config.attention_top_p_values[0]
    mp = config.masking_probabilities[0]
    salw = config.self_attention_loss_weights[0]
    top_k = next((value for value in config.attention_top_k_values if value is not None), None)

    cmd = build_base_command(config, override_mode="self_attention_oe") + [
        "--attention_generation_mode", "staged",
        "--attention_stage", "stage2",
        "--attention_top_p", f"{top_p}",
        "--masking_probability", f"{mp}",
        "--self_attention_loss_weight", f"{salw}",
        "--attention_cache_dir", str(config.attention_cache_base),
    ]
    if top_k is not None:
        cmd.extend(["--attention_top_k", str(top_k)])
    if config.stage3_standard_checkpoint:
        cmd.extend(["--stage3_standard_checkpoint", str(config.stage3_standard_checkpoint)])
    elif config.attention_cache_base:
        default_checkpoint = Path(config.attention_cache_base) / config.dataset / 'standard_model.pt'
        cmd.extend(["--stage3_standard_checkpoint", str(default_checkpoint)])
    if config.stage3_use_cached_standard_model:
        cmd.append("--stage3_use_cached_standard_model")

    print("[Stage2] Precomputing attention cache once before sweep...")
    start_time = datetime.now()
    returncode = run_command(cmd, stage2_log, env, stream_to_console=True)
    duration = datetime.now() - start_time
    print(f"  ↪ Stage2 finished (returncode={returncode}, duration={duration})")
    if returncode != 0:
        print(f"  ↪ Stage2 warmup failed, see {stage2_log}. Subsequent runs may error if cache is missing.")


def make_run_id(mode: str, generation_mode: str, stage: str | None, top_p: float,
                uniform_weight: float, mp: float, salw: float,
                top_k: int | None) -> str:
    """Generate unique run ID for experiment tracking (masking always enabled)"""
    stage_fragment = stage if stage else "na"
    top_k_fragment = "none" if top_k is None else f"k{top_k:02d}"
    # mask_fragment removed - masking is always enabled
    return (f"{mode}_{generation_mode}_stage{stage_fragment}_"
            f"{top_k_fragment}_topp{top_p:.2f}_uw{uniform_weight:.2f}_mp{mp:.2f}_salw{salw:.2f}")


def summarize_runs(summary_path: Path, completed_runs: Iterable[str]) -> None:
    if not completed_runs:
        print("\nNo successful runs to summarize.")
        return
    if not summary_path.exists():
        print("\nSummary file not found; skipping post-run summary.")
        return

    df = pd.read_csv(summary_path)
    df = df[df["run_id"].isin(completed_runs)].copy()
    if df.empty:
        print("\nNo matching entries in summary for the completed runs.")
        return

    pivot = pd.pivot_table(
        df[df["OE_Source"].str.contains("Self_Attention", na=False)],
        index=["run_id", "attention_generation_mode", "attention_stage"],  # stage3_masking_enabled removed
        values=["AUROC", "FPR95"],
        aggfunc={"AUROC": "max", "FPR95": "min"}
    ).reset_index()

    print("\n=== Sweep Summary (best Self-Attention metrics per run) ===")
    print(pivot.to_string(index=False))

    grouped = pivot.sort_values("AUROC", ascending=False)
    best_row = grouped.iloc[0]
    print("\n>>> Top configuration by AUROC:")
    print(best_row.to_string())


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "summary.csv"

    env = os.environ.copy()
    prepare_huggingface_cache(env)
    offline_requested = config.transformers_offline
    if config.transformers_offline and huggingface_cache_ready(env):
        env["TRANSFORMERS_OFFLINE"] = "1"
    elif config.transformers_offline:
        print("⚠️ HuggingFace cache missing required assets; running online to populate it.")
        config.transformers_offline = False
        env.pop("TRANSFORMERS_OFFLINE", None)
    else:
        env.pop("TRANSFORMERS_OFFLINE", None)

    maybe_run_stage2(config, env)

    if offline_requested and not config.transformers_offline and huggingface_cache_ready(env):
        print("✨ HuggingFace cache now populated; re-enabling offline mode for remaining runs.")
        config.transformers_offline = True
        env["TRANSFORMERS_OFFLINE"] = "1"

    completed_runs: list[str] = []

    for generation_mode in config.attention_generation_modes:
        stage_candidates = config.attention_stages if generation_mode == "staged" else [None]
        for stage in stage_candidates:
            # stage3_masking loop removed - masking is always enabled
            for top_k in config.attention_top_k_values:
                for top_p, mp, salw, uniform_weight in itertools.product(
                    config.attention_top_p_values,
                    config.masking_probabilities,
                    config.self_attention_loss_weights,
                    config.oe_uniform_loss_weights,
                ):
                    run_id = make_run_id(
                        config.mode,
                        generation_mode,
                        stage,
                        top_p,
                        uniform_weight,
                        mp,
                        salw,
                        top_k,
                        # stage3_masking removed
                    )
                    if offline_requested and not config.transformers_offline and huggingface_cache_ready(env):
                        print("✨ HuggingFace cache now populated; re-enabling offline mode for remaining runs.")
                        config.transformers_offline = True
                        env["TRANSFORMERS_OFFLINE"] = "1"
                    safe_run_id = run_id.replace('.', 'p')
                    log_path = config.output_dir / f"{safe_run_id}.log"
                    result_copy_path = config.output_dir / f"{safe_run_id}_results.csv"

                    if log_path.exists() and not config.overwrite:
                        print(f"[skip] {run_id}")
                        continue

                    cmd = build_base_command(config) + [
                        "--attention_generation_mode", generation_mode,
                    ]
                    if stage:
                        cmd.extend(["--attention_stage", stage])

                    cmd.extend([
                        "--attention_top_p", f"{top_p}",
                        "--masking_probability", f"{mp}",
                        "--self_attention_loss_weight", f"{salw}",
                        "--oe_uniform_loss_weight", f"{uniform_weight}",
                    ])

                    # --disable_attention_stage3_masking removed: masking is always enabled

                    if generation_mode == "staged" and config.attention_cache_base is not None:
                        cmd.extend(["--attention_cache_dir", str(config.attention_cache_base)])
                    if generation_mode == "staged" and config.stage3_use_cached_standard_model and stage == "stage3":
                        cmd.append("--stage3_use_cached_standard_model")
                    if generation_mode == "staged" and config.stage3_standard_checkpoint:
                        cmd.extend(["--stage3_standard_checkpoint", str(config.stage3_standard_checkpoint)])
                    elif generation_mode == "staged" and config.attention_cache_base is not None:
                        default_checkpoint = Path(config.attention_cache_base) / config.dataset / 'standard_model.pt'
                        cmd.extend(["--stage3_standard_checkpoint", str(default_checkpoint)])

                    if top_k is not None:
                        cmd.extend(["--attention_top_k", str(top_k)])

                    print(f"[run] {run_id}")
                    returncode = run_command(cmd, log_path, env)
                    if returncode != 0:
                        print(f"[fail] {run_id} (see {log_path})")
                        continue

                    try:
                        df = read_results(config.dataset)
                    except FileNotFoundError as exc:
                        print(f"[warn] {run_id}: {exc}")
                        continue

                    df = df.copy()
                    df["run_id"] = run_id
                    df["mode"] = config.mode
                    df["attention_generation_mode"] = generation_mode
                    df["attention_stage"] = stage if stage else "na"
                    df["attention_top_k"] = top_k if top_k is not None else "none"
                    df["attention_top_p"] = top_p
                    df["masking_probability"] = mp
                    df["self_attention_loss_weight"] = salw
                    df["oe_uniform_loss_weight"] = uniform_weight
                    # df["stage3_masking_enabled"] removed - masking is always enabled
                    df["stage3_cached_standard"] = bool(config.stage3_use_cached_standard_model and stage == "stage3")
                    if config.attention_cache_base and generation_mode == "staged":
                        df["attention_cache_dir"] = str(config.attention_cache_base)
                    df["timestamp"] = datetime.now().isoformat()

                    df.to_csv(result_copy_path, index=False)
                    append_summary(summary_path, df)
                    completed_runs.append(run_id)
                    print(f"[done] {run_id}")

    summarize_runs(summary_path, completed_runs)


if __name__ == "__main__":
    main()
