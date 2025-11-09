#!/usr/bin/env python3
"""
Simplified Outlier Exposure (OE) Pipeline with staged self-attention support.

This module implements research on generating hard negatives via global self-attention
masking while maintaining classic OE baselines. It provides:

- Baseline and traditional OE (external datasets) for comparison baselines.
- Self-attention OE with Hendrycks-style simultaneous ID+OE training.
- Staged workflow (Stage2 + Stage3) for efficient hyperparameter sweeps.

=== Hendrycks-Style Training Architecture ===

Following Hendrycks et al. (2019), this implementation trains models with:
  Loss = CE(x_in) + λ * L_uniform(x_out)

where ID and OE batches are processed simultaneously via zip/cycle pattern,
NOT sequentially (ID first, then OE fine-tuning).

=== Three Execution Modes ===

**Mode 1: Stage2 Only (Attention Caching)**
  Purpose: Cache attention metadata for Stage3 reuse
  - Trains Standard model (ID data only) from pretrained RoBERTa
  - Extracts attention scores for all ID training samples
  - Saves attention metadata to disk (sharded for efficiency)
  - NO OE training (caching only)

  Example:
  ```bash
  python scripts/run_oe_sweep.py --dataset 20newsgroups --mode self_attention_oe \
      --attention_generation_modes staged --attention_stages stage2 \
      --attention_top_p_values 0.15,0.20,0.25 \
      --masking_probabilities 0.05 \
      --oe_uniform_loss_weights 1.0 \
      --attention_cache_base simplified_oe_experiments/oe_cache \
      --output_dir sweeps/oe/stage2
  ```

  Output: Cache at {cache_base}/{dataset}/{algorithm}/topp{value}/stage2/
    - metadata.json (configuration and sample count)
    - shard_*.pt files (attention scores and spans)

**Mode 2: Stage3 Only (Hendrycks-Style, RECOMMENDED)**
  Purpose: Efficient hyperparameter sweep with fresh initialization
  - Prerequisite: Stage2 cache must exist
  - Standard model: Fresh pretrained RoBERTa
  - OE models: **Fresh pretrained RoBERTa (Hendrycks-style)**
  - Uses CachedAttentionOEDataset for on-the-fly OE generation
  - Trains with zip(ID_loader, cycle(OE_loader))
  - NO Standard checkpoint loading → prevents ID overfitting

  Example:
  ```bash
  python scripts/run_oe_sweep.py --dataset 20newsgroups --mode self_attention_oe \
      --attention_generation_modes staged --attention_stages stage3 \
      --attention_top_p_values 0.15,0.20,0.25 \
      --masking_probabilities 0.05,0.10,0.15 \
      --oe_uniform_loss_weights 1.0 \
      --attention_cache_base simplified_oe_experiments/oe_cache \
      --output_dir sweeps/oe/stage3 \
      --no_stage2_warmup \
      --overwrite
  ```

  Key Features:
    ✅ Fresh initialization (Hendrycks principle)
    ✅ On-the-fly OE generation (memory efficient)
    ✅ Simultaneous ID+OE training
    ✅ Expected +0.5~2.0% AUROC improvement over sequential training

**Mode 3: Stage2+Stage3 Together (Legacy, Backward Compatibility)**
  Purpose: One-shot pipeline (less efficient, sequential training)
  - Trains Standard model
  - Caches attention metadata
  - OE models: **Load Standard checkpoint (sequential training)**
  - Uses OSRTextDataset (in-memory materialization)
  - Higher memory usage, lower performance

  Example:
  ```bash
  python scripts/run_oe_sweep.py --dataset 20newsgroups --mode self_attention_oe \
      --attention_generation_modes staged --attention_stages both \
      --attention_top_p_values 0.15,0.20 \
      --masking_probabilities 0.05,0.10 \
      --oe_uniform_loss_weights 1.0 \
      --attention_cache_base simplified_oe_experiments/oe_cache \
      --output_dir sweeps/oe/both \
      --overwrite
  ```

  ⚠️ Warning: This mode uses sequential training (ID → OE fine-tuning),
              which violates Hendrycks' simultaneous training principle.
              Use Mode 2 (Stage3 only) for best performance.

=== Key Implementation Details ===

1. **Fresh Initialization (Hendrycks Requirement)**
   - Stage3-only mode: OE models start from pretrained RoBERTa
   - Stage2+Stage3 mode: OE models start from Standard checkpoint (legacy)
   - Controlled by `stage3_only_request` flag (oe.py:3831-3834)

2. **CachedAttentionOEDataset (On-the-fly OE)**
   - Loads attention metadata from Stage2 cache
   - Applies masking dynamically during training
   - Minimal memory footprint (lazy loading of shards)
   - Implementation: oe.py:1455-1590

3. **Hendrycks zip/cycle Pattern**
   - CombinedOSRDataLoader yields (ID_batch, OE_batch) pairs
   - OE data cycles when exhausted
   - Implementation: oe.py:1346-1368

4. **Uniform Loss**
   - Formula: -log_softmax(logits).mean()
   - Weight: λ=1.0 (Hendrycks default)
   - Implementation: oe.py:1957-1962

5. **Cache Path Resolution**
   - Centralized functions: format_float_for_path(), resolve_attention_cache_path()
   - Consistent path format: {base}/{dataset}/{algorithm}/topp{value}/
   - Implementation: oe.py:462-521

=== Removed Features (2025-10-09) ===

- ❌ STAGE3_MASKING_OPTIONS: Masking is always enabled (Hendrycks requirement)
- ❌ --disable_attention_stage3_masking: Flag removed
- ❌ stage3_masking ablation: Simplified to single masking approach

=== Recommended Workflow ===

1. Run Stage2 once to cache attention metadata
2. Run Stage3 multiple times with different hyperparameters
3. Use --no_stage2_warmup to skip Stage2 warmup in Stage3 runs
4. Compare AUROC: Stage3-only should outperform Stage2+Stage3 by +0.5~2%

This design enables:
  ✅ Efficient hyperparameter sweeps (Stage2 once, Stage3 many times)
  ✅ Faithful Hendrycks simultaneous training (Stage3-only mode)
  ✅ Minimal memory footprint (on-the-fly OE generation)
  ✅ Backward compatibility (Stage2+Stage3 mode preserved)
"""

import os
import re
import sys
import json
import math
import argparse
from types import SimpleNamespace
import warnings
import logging
import random
import hashlib
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass
from pathlib import Path

# Configure environment for cleaner logging output
os.environ['TQDM_NCOLS'] = '80'   # Fixed width for consistent formatting

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor

# Enable SDPA (Scaled Dot Product Attention) for memory efficiency
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WandbLogger = None  # type: ignore
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False
    print("⚠️  WandB not available, using basic logging")

# TensorBoard removed - using WandB only
import torchmetrics
import torchmetrics.functional as tmf

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Configuration Class
# =============================================================================

class Config:
    """
    Simplified configuration for Outlier Exposure experiments
    Removed complex attention-based parameters and simplified to core OE functionality
    """
    
    # === Basic Experiment Settings ===
    CURRENT_NLP_DATASET = "ag_news"  # Current NLP dataset to use
    
    # === Model and Training Settings ===
    MODEL_NAME = "roberta-base"  # Base model name
    MAX_LENGTH = 256   # Maximum text length (tokens)
    BATCH_SIZE = 64  # Training batch size
    EVAL_BATCH_SIZE = 64  # Evaluation batch size
    NUM_EPOCHS = 7  # Number of training epochs
    LEARNING_RATE = 2e-5  # Learning rate
    NUM_WORKERS = 0  # DataLoader worker count (0 for memory optimization)
    RANDOM_STATE = 42  # Random seed for reproducibility
    BASE_BATCH_SIZE = 64  # Reference batch size for LR scaling
    BASE_LEARNING_RATE = 2e-5  # Reference learning rate for LR scaling
    MAX_LEARNING_RATE = 5e-5  # Safety ceiling when auto-scaling
    
    # === Data Processing Settings ===
    TEXT_COLUMN = "text"  # Text column name
    EXCLUDE_CLASS_FOR_TRAINING = "unknown"  # Class to exclude from training
    MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL = 2  # Minimum samples per class for train/val
    USE_WEIGHTED_LOSS = True  # Use class-weighted loss
    
    # === Output and Storage Settings ===
    OUTPUT_DIR = 'simplified_oe_experiments'  # Experiment results directory
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")  # Model save directory
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")  # Logging directory
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")  # Results directory
    
    # === OSR (Open Set Recognition) Settings ===
    OSR_EXPERIMENT_METHOD = 'cross'  # 'cross' | 'holdout' - OSR experiment method
    OSR_HOLDOUT_RATIO = 0.2  # Fraction of classes to hold out for OOD
    OSR_HOLDOUT_MIN_CLASSES = 1  # Minimum number of classes to hold out
    OSR_MAX_LENGTH = 256  # Maximum sequence length for OSR
    OSR_BATCH_SIZE = 64  # Batch size for OSR
    OSR_LEARNING_RATE = 2e-5  # Learning rate for OSR
    OSR_NUM_EPOCHS = 5  # Number of epochs for OSR (reduced for faster training)
    OSR_EARLY_STOPPING_PATIENCE = 2  # Early stopping patience
    OSR_EARLY_STOPPING_MIN_DELTA = 0.001  # Early stopping minimum delta
    OSR_NUM_DATALOADER_WORKERS = 0  # Number of dataloader workers (0 for memory optimization)
    OSR_MODEL_DIR = os.path.join(OUTPUT_DIR, "osr_models")  # OSR model directory

    # === External OE Dataset Settings ===
    # OE Datasets for training (Cross-dataset scenario)
    OSR_EXTERNAL_OE_DATASETS = [
        'wikitext', 'ag_news',  # Removed wmt16 due to loading issues
    ]
    # OSR_EXTERNAL_OE_DATASETS = [
    #     'wikitext', 'ag_news', 'wmt16', 'bbc_news', 'imdb'
    # ]
        
    # === OE Method Settings (Dynamic based on mode) ===
    OE_METHOD = "traditional"  # "traditional", "self_attention", or "both"
    USE_SELF_ATTENTION_OE = False  # Enable Self Attention-guided OE (set dynamically)
    USE_TRADITIONAL_OE = True  # Enable Traditional OE (set dynamically)
    
    # === Self Attention-guided OE Settings ===
    ATTENTION_TOP_K = 3  # Top-K tokens to mask based on attention
    ATTENTION_TOP_P = 0.15  # Cumulative attention ratio for token selection
    MIN_ATTENTION_TOP_TOKENS = 3  # Minimum tokens to retain after filtering
    USE_ATTENTION_ELBOW_REFINEMENT = True  # Apply elbow after top-p selection
    MASKING_PROBABILITY = 0.3  # Probability to mask important tokens
    HARD_NEGATIVE_RATIO = 0.0  # Deprecated: kept for backwards compat, no longer used
    OE_UNIFORM_LOSS_WEIGHT = 1.0  # Weight for uniform OE loss component
    OE_ID_TO_OOD_RATIO = 1  # Deprecated: kept for backwards compat (no longer enforces ID:OE ratio)
    SELF_ATTENTION_LOSS_WEIGHT = 1.0  # Weight for self-attention hard negative loss
    ATTENTION_LAYER_INDEX = -1  # Which layer to extract attention (-1 for last layer)
    ATTENTION_FILTERING_METHOD = "basic"  # "basic", "entropy_elbow_higher", etc.
    MASK_TOKEN_ID = 50264  # RoBERTa mask token [MASK]
    ATTENTION_GENERATION_MAX_SAMPLES = None  # Max samples for pre-generated OE data (None -> no extra cap)
    OE_MAX_SAMPLES = None  # Maximum OE samples per scenario (None -> match ID sample count)
    OE_SAMPLING_SEED = 42  # Seed for OE sampling reproducibility
    OOD_EVAL_MAX_SAMPLES = 3000  # Max OOD samples per dataset during evaluation
    ENABLE_ATTENTION_DEBUG_LOGS = False  # Verbose debug for attention token selection
    ATTENTION_DEBUG_MAX_EXAMPLES = 5  # How many debug entries per category
    DISABLE_MODEL_CHECKPOINT = False  # Skip saving lightning checkpoints when running sweeps
    USE_WORD_LEVEL_ATTENTION = True  # Aggregate token scores to word spans before selection
    USE_ATTENTION_IDF_PENALTY = False  # Apply IDF-style down-weighting to frequent words

    # === Attention generation modes ===
    ATTENTION_GENERATION_MODE = "on_the_fly"  # "on_the_fly" or "staged"
    ATTENTION_STAGE_TO_RUN = "both"  # "stage2", "stage3", or "both"
    ATTENTION_CACHE_DIR = os.path.join(OUTPUT_DIR, "oe_cache")
    ATTENTION_STAGE2_SHARD_SIZE = 512  # Number of samples per shard saved during stage2
    # ATTENTION_STAGE3_APPLY_MASKING removed - masking is always enabled (Hendrycks requirement)
    STAGE3_USE_CACHED_STANDARD_MODEL = False  # Reuse saved Standard checkpoint when running staged stage3 only
    STAGE3_STANDARD_CHECKPOINT = None  # Optional path to saved Standard model state dict
    ENABLE_DEVICE_STATS_MONITOR = False  # Toggle PyTorch Lightning DeviceStatsMonitor callback
    
    # === NLP Dataset Definitions ===
    # === Experiment Design ===
    # In-Distribution (ID) Datasets

    # Out-of-Distribution (OOD) Datasets for Cross-dataset scenario
    # 데이터셋	Train	Dev/Val	Test	Total	비고
    # 20 Newsgroups	11,314	—	7,532	18,846	‘bydate’ 표준 split 기준임
    # TREC-6	5,452	—	500	5,952	상위 6개 클래스 버전 기준임
    # SST-2 (GLUE)	67,349	872	1,821	70,042	문장 단위 감성 분류임
    # AG_NEWS	120,000	—	7,600	127,600	4클래스(Each 30k/1.9k)임
    # WikiText-2	600 문서	60 문서	60 문서	720 문서	LM용이라 “샘플=문서”. 토큰: ≈2.0M/0.22M/0.25M
    # WikiText-103	28,475 문서	60 문서	60 문서	28,595 문서	토큰: ≈103M/≈0.22M/≈0.25M 임
    # WMT16 En–De	~4.5M 쌍	newsdev2016 (~3k)	newstest2016 (~3k)	~4.506M+	문장쌍 기준. 코퍼스 구성에 따라 약간 변동됨
    # WMT16 En–Ro	~610k 쌍	newsdev2016 (~2k)	newstest2016 (~2k)	~614k+	En–Ro 소규모 벤치마크로 자주 사용됨
    
    
    #   1. WMT16: 4.5M개 (번역 데이터셋)
    #   2. AG News: 127K개
    #   3. SST2: 70K개
    #   4. WikiText-2: 44K개
    #   5. 20newsgroups: 18K개
    
    
    
    #       - SST-2 (GLUE/sst2)
    #       - train: 67,349
    #       - validation: 872
    #       - test: 1,821
    #   - 20 Newsgroups (SetFit/20_newsgroups)
    #       - train: 11,314
    #       - test: 7,532
    #   - WikiText-2 raw
    #       - train: 36,718
    #       - validation: 3,760
    #       - test: 4,358
    #   - WMT16 (de-en)
    #       - train: 4,548,885
    #       - validation: 2,169
    #       - test: 2,999
    NLP_DATASETS = {
        # ID Datasets
        'sst2': {'name': 'glue', 'subset': 'sst2', 'text_column': 'sentence', 'label_column': 'label'},
        'trec': {'name': 'trec', 'subset': None, 'text_column': 'text', 'label_column': 'fine_label', 'num_classes': 50, 'trust_remote_code': True},
        '20newsgroups': {'name': 'SetFit/20_newsgroups', 'subset': None, 'text_column': 'text', 'label_column': 'label'},
        
        # OOD Datasets for Cross-dataset scenario
        'ag_news': {'name': 'ag_news', 'subset': None, 'text_column': 'text', 'label_column': 'label'},
        'wikitext': {'name': 'wikitext', 'subset': 'wikitext-2-raw-v1', 'text_column': 'text', 'label_column': None},
        'wmt16': {'name': 'wmt16', 'subset': 'de-en', 'text_column': 'en', 'label_column': None},
        
        # Additional datasets for OE
        'bbc_news': {'name': 'SetFit/bbc-news', 'subset': None, 'text_column': 'text', 'label_column': 'label'},
        'imdb': {'name': 'imdb', 'subset': None, 'text_column': 'text', 'label_column': 'label'},
    }
    
    # === Hardware Settings ===
    HUGGINGFACE_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/transformers")
    
    # WandB Configuration
    WANDB_API_KEY = ""  # Default API key
    WANDB_PROJECT = "20251009-NEWSGROUP-4090"  # Project name
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.OUTPUT_DIR,
            cls.MODEL_DIR,
            cls.LOG_DIR,
            cls.RESULTS_DIR,
            cls.OSR_MODEL_DIR,
            cls.ATTENTION_CACHE_DIR,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print(f"Created directories: {directories}")
    
    @classmethod
    def setup_for_dataset(cls, dataset_name: str):
        """Setup configuration for specific dataset - using centralized directories"""
        # Use centralized directories instead of dataset-specific ones
        cls.create_directories()
        print(f"Setup for dataset: {dataset_name} (using centralized directories)")
        
    @classmethod
    def set_oe_mode(cls, mode: str):
        """Set OE method based on experiment mode"""
        if mode == "baseline":
            cls.OE_METHOD = "baseline"
            cls.USE_SELF_ATTENTION_OE = False
            cls.USE_TRADITIONAL_OE = False
            print("🔧 Mode: Baseline (No OE)")
            
        elif mode == "traditional_oe":
            cls.OE_METHOD = "traditional"
            cls.USE_SELF_ATTENTION_OE = False
            cls.USE_TRADITIONAL_OE = True
            print("🔧 Mode: Traditional OE (External datasets)")
            
        elif mode == "self_attention_oe":
            cls.OE_METHOD = "self_attention" 
            cls.USE_SELF_ATTENTION_OE = True
            cls.USE_TRADITIONAL_OE = False
            print(f"🔧 Mode: Self Attention-guided OE (Method: {cls.ATTENTION_FILTERING_METHOD})")
            
        elif mode == "both_oe":
            cls.OE_METHOD = "both"
            cls.USE_SELF_ATTENTION_OE = True
            cls.USE_TRADITIONAL_OE = True
            print("🔧 Mode: Both OE methods (Traditional + Self Attention)")
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: baseline, traditional_oe, self_attention_oe, both_oe")

    @classmethod
    def adjust_learning_rates_for_batch_size(cls, auto_adjust: bool = True):
        """Scale learning rates in proportion to batch size when auto-adjust is enabled."""
        if not auto_adjust:
            return

        batch_size = max(int(cls.BATCH_SIZE), 1)
        reference = max(int(cls.BASE_BATCH_SIZE), 1)
        if batch_size == reference:
            return

        scale_factor = batch_size / reference
        scaled_lr = cls.BASE_LEARNING_RATE * scale_factor

        if cls.MAX_LEARNING_RATE:
            scaled_lr = min(scaled_lr, cls.MAX_LEARNING_RATE)

        cls.LEARNING_RATE = scaled_lr
        cls.OSR_LEARNING_RATE = scaled_lr

        print(
            f"⚙️  Auto-scaled learning rate for batch_size={batch_size}: "
            f"{scaled_lr:.2e} (scale={scale_factor:.2f}, base={cls.BASE_LEARNING_RATE:.2e})"
        )

# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

def setup_logging(output_dir: str, dataset_name: str, experiment_type: str = "experiment") -> logging.Logger:
    """Setup logging to both file and console with timestamp and command logging"""
    
    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_type}_{dataset_name}_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Create logger
    logger = logging.getLogger(f"oe_experiment_{dataset_name}")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters with enhanced timestamp
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler  
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log initial message with timestamp and command
    logger.info(f"=== Starting {experiment_type} for {dataset_name} ===")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logger.info(f"Execution command: {' '.join(sys.argv)}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log file: {log_filepath}")
    
    return logger

def preprocess_text_for_roberta(text: str) -> str:
    """Simple text preprocessing for RoBERTa"""
    if not isinstance(text, str):
        return ""
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = text.strip()
    return text


def format_wandb_metric_key(prefix: str, name: str, metric: str) -> str:
    """Sanitize metric keys so each OOD dataset renders as a separate WandB series."""
    sanitized = re.sub(r'[^0-9A-Za-z_]+', '_', name)
    if not sanitized:
        sanitized = "unknown"
    if sanitized[0].isdigit():
        sanitized = f"dataset_{sanitized}"
    return f"{prefix}/{sanitized}/{metric}"


def safe_float_for_wandb(value: Optional[float]) -> Optional[float]:
    """Convert values to floats for WandB tables, dropping NaNs to avoid serialization errors."""
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def format_float_for_path(value: Optional[float]) -> str:
    """
    Format float value for use in file paths.

    Args:
        value: Float value to format, or None

    Returns:
        Formatted string with trailing zeros and decimal point removed

    Examples:
        0.15 -> "0.15"
        0.20 -> "0.2"
        1.00 -> "1"
        None -> "none"
    """
    if value is None:
        return "none"
    formatted = f"{float(value):.2f}".rstrip('0').rstrip('.')
    return formatted if formatted else "0"


def resolve_attention_cache_path(
    base_dir: Path,
    dataset: str,
    algorithm: str,
    top_p: Optional[float]
) -> Path:
    """
    Resolve attention cache directory path with consistent naming convention.

    This function provides a centralized way to construct cache paths used by both
    oe.py and run_oe_sweep.py, ensuring consistency across the codebase.

    Args:
        base_dir: Base cache directory (e.g., simplified_oe_experiments/oe_cache)
        dataset: Dataset name (e.g., "20newsgroups")
        algorithm: Attention algorithm name (e.g., "entropy_elbow_higher")
        top_p: Attention top-p threshold value

    Returns:
        Complete cache directory path

    Example:
        resolve_attention_cache_path(
            Path("cache"), "20newsgroups", "entropy_elbow_higher", 0.15
        )
        -> Path("cache/20newsgroups/entropy_elbow_higher/topp0.15")
    """
    cache_root = base_dir / dataset / algorithm

    if top_p is not None:
        top_p_fragment = f"topp{format_float_for_path(top_p)}"
        return cache_root / top_p_fragment
    else:
        return cache_root


def tensor_to_float(value: Optional[Any], default: float = float('nan')) -> float:
    """Best-effort conversion from tensors/arrays to primitive floats for logging."""
    if value is None:
        return default
    try:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        if hasattr(value, 'item'):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError):
        return default


# =============================================================================
# Self Attention-guided OE Components
# =============================================================================

class SelfAttentionAnalyzer:
    """
    Analyzes self-attention patterns to identify important tokens for generating hard negatives
    """
    
    def __init__(self, config: Config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.mask_token_id = config.MASK_TOKEN_ID
        self.debug_enabled = bool(getattr(config, 'ENABLE_ATTENTION_DEBUG_LOGS', False))
        self.debug_max_examples = max(int(getattr(config, 'ATTENTION_DEBUG_MAX_EXAMPLES', 0)), 0)
        self._debug_counters: Dict[str, int] = {}
        self._debug_logger = logging.getLogger(f"oe_attention_{config.CURRENT_NLP_DATASET}")
        if self.debug_enabled:
            self._debug_logger.setLevel(logging.DEBUG)
            if not any(isinstance(handler, logging.StreamHandler) for handler in self._debug_logger.handlers):
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                stream_handler.setFormatter(formatter)
                self._debug_logger.addHandler(stream_handler)
            self._debug_logger.propagate = False
        # Track tokenizer-specific special tokens for word span handling
        special_tokens: Set[int] = set()
        for attr in [
            'bos_token_id', 'eos_token_id', 'cls_token_id', 'sep_token_id', 'pad_token_id',
            'unk_token_id', 'mask_token_id'
        ]:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                if isinstance(token_id, list):
                    special_tokens.update(token_id)
                else:
                    special_tokens.add(token_id)
        self.special_token_ids = special_tokens
        # Word-level aggregation & IDF control flags
        self.use_word_level = bool(getattr(config, 'USE_WORD_LEVEL_ATTENTION', True))
        self.use_idf_penalty = bool(getattr(config, 'USE_ATTENTION_IDF_PENALTY', False))
        self._word_document_frequency: Dict[str, int] = {}
        self.total_documents = 0

    def _should_log_debug(self, category: str) -> bool:
        if not self.debug_enabled or self.debug_max_examples <= 0:
            return False
        count = self._debug_counters.get(category, 0)
        if count >= self.debug_max_examples:
            return False
        self._debug_counters[category] = count + 1
        return True

    def _log_debug(self, message: str):
        self._debug_logger.debug(message)

    def _normalize_word(self, text: str) -> str:
        """Normalize word text for IDF bookkeeping."""
        if not text:
            return ""
        normalized = text.strip().lower()
        return normalized

    def _maybe_log_filter_summary(
        self,
        category: str,
        batch_idx: int,
        total_tokens: int,
        selected_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._should_log_debug(category):
            return
        extra_segments = []
        if extra:
            extra_segments = [f"{key}={value}" for key, value in extra.items()]
        extra_payload = (", " + ", ".join(extra_segments)) if extra_segments else ""
        self._log_debug(
            f"[{category}] batch={batch_idx} total_tokens={total_tokens} selected={selected_tokens}{extra_payload}"
        )

    def compute_token_scores(
        self,
        algorithm: str,
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[List[int]]], List[List[str]]]:
        """Return per-sample word scores, indices, spans, and word texts for the given algorithm."""

        algorithm = algorithm.lower()
        batch_size = attention_weights.size(0)

        scores_per_sample: List[torch.Tensor] = []
        indices_per_sample: List[torch.Tensor] = []
        spans_per_sample: List[List[List[int]]] = []
        words_per_sample: List[List[str]] = []

        if algorithm == 'entropy_elbow_higher':
            base_scores = self.calculate_attention_entropy(attention_weights)
            while base_scores.dim() > 2:
                base_scores = base_scores.mean(dim=1)
        elif algorithm == 'max_attention_elbow_lower':
            base_scores = attention_weights.max(dim=-1)[0]
            while base_scores.dim() > 2:
                base_scores = base_scores.mean(dim=1)
        elif algorithm == 'removed_avg_elbow_higher':
            base_scores = attention_weights.sum(dim=-1)
            while base_scores.dim() > 2:
                base_scores = base_scores.mean(dim=1)
        elif algorithm == 'top_k_avg_elbow_lower':
            top_k = max(int(getattr(self.config, 'ATTENTION_TOP_K', 1)), 1)
            top_k = min(top_k, attention_weights.size(-1))
            base_scores = attention_weights.topk(top_k, dim=-1)[0].mean(dim=-1)
            while base_scores.dim() > 2:
                base_scores = base_scores.mean(dim=1)
        else:
            raise ValueError(f"Unsupported algorithm for score computation: {algorithm}")

        for batch_idx in range(batch_size):
            mask = self._batch_mask(attention_mask, batch_idx)
            valid_indices = torch.nonzero(mask, as_tuple=False).flatten()

            if valid_indices.numel() == 0:
                scores_per_sample.append(torch.tensor([], device=attention_weights.device))
                indices_per_sample.append(torch.tensor([], dtype=torch.long, device=attention_weights.device))
                spans_per_sample.append([])
                words_per_sample.append([])
                continue

            sample_scores = base_scores[batch_idx]
            if sample_scores.dim() == 0:
                sample_scores = sample_scores.repeat(mask.size(0))

            valid_scores = sample_scores[mask]

            if algorithm == 'removed_avg_elbow_higher' and valid_scores.numel() > 0:
                avg_importance = valid_scores.mean()
                valid_scores = valid_scores - avg_importance

            # Build token -> word spans
            token_ids = input_ids[batch_idx].tolist()
            token_strings = self.tokenizer.convert_ids_to_tokens(token_ids)
            token_to_valid_idx = {int(tok_idx): pos for pos, tok_idx in enumerate(valid_indices.tolist())}

            word_spans: List[List[int]] = []
            word_texts: List[str] = []

            def is_start(token: str, position: int) -> bool:
                if position == 0:
                    return True
                if token.startswith('Ġ') or token.startswith('▁') or token.startswith('Ċ'):
                    return True
                return False

            current_span: List[int] = []
            for token_pos, token in enumerate(token_strings):
                if token_pos >= mask.numel() or not bool(mask[token_pos]):
                    continue
                if token_ids[token_pos] in self.special_token_ids:
                    continue
                if not current_span or is_start(token, token_pos):
                    if current_span:
                        word_spans.append(current_span)
                        word_texts.append(self.tokenizer.convert_tokens_to_string([token_strings[i] for i in current_span]))
                    current_span = [token_pos]
                else:
                    current_span.append(token_pos)
            if current_span:
                word_spans.append(current_span)
                word_texts.append(self.tokenizer.convert_tokens_to_string([token_strings[i] for i in current_span]))

            word_scores: List[float] = []
            filtered_spans: List[List[int]] = []
            filtered_words: List[str] = []
            doc_words: Set[str] = set()
            current_total_docs = max(self.total_documents, 1)
            for span, text in zip(word_spans, word_texts):
                valid_positions = [token_to_valid_idx[idx] for idx in span if idx in token_to_valid_idx]
                if not valid_positions:
                    continue
                span_scores = valid_scores[valid_positions]
                score_value = float(span_scores.mean().item())
                if self.use_idf_penalty:
                    normalized = self._normalize_word(text)
                    if normalized:
                        doc_freq = self._word_document_frequency.get(normalized, 0)
                        idf_weight = math.log((1 + current_total_docs) / (1 + doc_freq)) + 1.0
                        score_value *= idf_weight
                        doc_words.add(normalized)
                word_scores.append(score_value)
                filtered_spans.append(span)
                filtered_words.append(text.strip())

            if len(word_scores) == 0:
                scores_per_sample.append(torch.tensor([], device=attention_weights.device))
                indices_per_sample.append(torch.tensor([], dtype=torch.long, device=attention_weights.device))
                spans_per_sample.append([])
                words_per_sample.append([])
                continue

            scores_tensor = torch.tensor(word_scores, dtype=valid_scores.dtype, device=valid_scores.device)
            scores_per_sample.append(scores_tensor)
            indices_per_sample.append(torch.arange(len(word_scores), device=valid_scores.device, dtype=torch.long))
            spans_per_sample.append(filtered_spans)
            words_per_sample.append(filtered_words)
            if self.use_idf_penalty and doc_words:
                self.total_documents += 1
                for norm_word in doc_words:
                    self._word_document_frequency[norm_word] = self._word_document_frequency.get(norm_word, 0) + 1
            elif self.use_idf_penalty:
                self.total_documents += 1

        return scores_per_sample, indices_per_sample, spans_per_sample, words_per_sample

    def compute_global_threshold(
        self,
        all_scores: torch.Tensor,
        higher: bool,
        context: str,
    ) -> Tuple[Optional[float], int]:
        """Compute global threshold using top-p followed by elbow on aggregated scores."""

        if all_scores.numel() == 0:
            return None, 0

        device = all_scores.device
        global_indices = torch.arange(all_scores.numel(), device=device)
        selected_indices = self._select_top_p_then_elbow(
            all_scores,
            global_indices,
            higher=higher,
            context=f"global_{context}"
        )

        if selected_indices.numel() == 0:
            if higher:
                threshold_value = float(all_scores.max().item())
            else:
                threshold_value = float(all_scores.min().item())
            return threshold_value, 0

        selected_scores = all_scores[selected_indices]
        if higher:
            threshold_value = float(selected_scores.min().item())
        else:
            threshold_value = float(selected_scores.max().item())

        return threshold_value, int(selected_indices.numel())

    def _batch_mask(self, attention_mask: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Return boolean mask for a batch index, accepting 1D or 2D masks."""
        if attention_mask.dim() == 1:
            return attention_mask.bool()
        if attention_mask.dim() == 2:
            return attention_mask[batch_idx].bool()
        raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

    def extract_attention_weights(self, model, input_ids, attention_mask):
        """Extract attention weights from specified layer"""
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Get attention from specified layer
            attention_weights = outputs.attentions[self.config.ATTENTION_LAYER_INDEX]
            # Average across heads: [batch_size, seq_len, seq_len]
            attention_weights = attention_weights.mean(dim=1)
            
        return attention_weights
    
    def calculate_attention_entropy(self, attention_weights):
        """Calculate entropy of attention distribution for each position"""
        # Normalize attention weights to get probabilities
        attention_probs = F.softmax(attention_weights, dim=-1)
        # Calculate entropy: -sum(p * log(p))
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
        return entropy
    
    def find_elbow_point(self, values, higher=True):
        """Find elbow point in sorted values using gradient method"""
        sorted_values, _ = torch.sort(values, descending=higher)
        
        # Calculate first and second derivatives
        if len(sorted_values) < 3:
            return len(sorted_values) // 2
            
        first_diff = sorted_values[1:] - sorted_values[:-1]
        second_diff = first_diff[1:] - first_diff[:-1]
        
        # Find elbow as point of maximum curvature
        curvature = torch.abs(second_diff)
        elbow_idx = torch.argmax(curvature).item() + 1
        
        return min(elbow_idx, len(sorted_values) - 1)

    def _select_top_p_then_elbow(
        self,
        valid_scores: torch.Tensor,
        valid_indices: torch.Tensor,
        higher: bool = True,
        context: str = "unknown"
    ) -> torch.Tensor:
        """Apply top-p selection (with minimum token guard) then optional elbow refinement."""

        if valid_scores.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=valid_indices.device)

        top_p = float(getattr(self.config, 'ATTENTION_TOP_P', 0.15) or 0.0)
        top_p = max(0.0, min(1.0, top_p))
        min_tokens = max(int(getattr(self.config, 'MIN_ATTENTION_TOP_TOKENS', 1)), 1)
        use_elbow = bool(getattr(self.config, 'USE_ATTENTION_ELBOW_REFINEMENT', True))

        sorted_scores, sorted_order = torch.sort(valid_scores, descending=higher)

        # Derive positive weights so cumulative mass emphasizes desired direction
        anchor = sorted_scores[-1]
        if higher:
            weights = sorted_scores - anchor
        else:
            weights = anchor - sorted_scores
        weights = torch.clamp(weights, min=0.0)
        weights_sum = torch.sum(weights)
        if weights_sum.item() <= 0:
            weights = torch.ones_like(sorted_scores)

        cumulative = torch.cumsum(weights, dim=0)
        total_value = float(cumulative[-1].item())

        if top_p <= 0.0:
            top_count = min_tokens
        elif top_p >= 1.0 or total_value <= 0.0:
            top_count = len(sorted_scores)
        else:
            normalized = cumulative / total_value
            ratio_tensor = torch.tensor(top_p, device=normalized.device, dtype=normalized.dtype)
            top_count = torch.searchsorted(normalized, ratio_tensor, right=True).item() + 1
            top_count = max(min_tokens, top_count)
        pre_elbow_count = top_count

        top_count = min(top_count, len(sorted_scores))
        selected_order = sorted_order[:top_count]
        selected_scores = sorted_scores[:top_count]

        elbow_idx_value: Optional[int] = None
        threshold_value_export: Optional[float] = None
        elbow_applied = False

        if use_elbow and len(selected_scores) >= 3:
            elbow_idx = self.find_elbow_point(selected_scores, higher=higher)
            elbow_idx = min(max(elbow_idx, 0), len(selected_scores) - 1)
            threshold_tensor = torch.sort(selected_scores, descending=higher)[0][elbow_idx]
            if higher:
                keep_mask = selected_scores >= threshold_tensor
            else:
                keep_mask = selected_scores <= threshold_tensor
            refined_order = selected_order[keep_mask]
            if len(refined_order) >= min_tokens:
                selected_order = refined_order
                elbow_idx_value = int(elbow_idx)
                threshold_value_export = float(threshold_tensor.detach().cpu().item())
                elbow_applied = True

        if len(selected_order) < min_tokens:
            fallback_count = min(min_tokens, len(sorted_scores))
            selected_order = sorted_order[:fallback_count]
            fallback_triggered = True
        else:
            fallback_triggered = False

        final_count = len(selected_order)

        if self._should_log_debug(f"selection_{context}"):
            preview_limit = min(5, len(sorted_scores))
            sorted_preview = (
                sorted_scores[:preview_limit].detach().cpu().tolist()
                if preview_limit > 0
                else []
            )
            cumulative_preview = []
            if top_p > 0.0 and top_p < 1.0 and total_value > 0.0:
                cumulative_preview = (
                    (cumulative / total_value)[:preview_limit].detach().cpu().tolist()
                )
            final_preview_limit = min(5, final_count)
            final_scores = (
                valid_scores[selected_order][:final_preview_limit].detach().cpu().tolist()
                if final_count > 0 and final_preview_limit > 0
                else []
            )
            debug_message = (
                f"[select_top_p_then_elbow][{context}] top_p={top_p:.4f}, higher={higher}, "
                f"candidates={len(valid_scores)}, min_tokens={min_tokens}, total_value={total_value:.6f}, "
                f"pre_elbow={pre_elbow_count}, post_elbow={final_count}, elbow_applied={elbow_applied}, "
                f"elbow_idx={elbow_idx_value}, threshold={threshold_value_export}, fallback={fallback_triggered}, "
                f"sorted_head={sorted_preview}, cumulative_head={cumulative_preview}, final_head={final_scores}"
            )
            self._log_debug(debug_message)

        return valid_indices[selected_order]
    
    def attention_entropy_elbow_higher(self, attention_weights, attention_mask):
        """Filter tokens with higher entropy using elbow method"""
        entropy = self.calculate_attention_entropy(attention_weights)

        # Check entropy dimensions and adjust accordingly
        if entropy.dim() > 2:
            # If entropy has more than 2 dims, average across heads/layers
            while entropy.dim() > 2:
                entropy = entropy.mean(dim=1)

        # avg_entropy should be [batch_size, seq_len]
        if entropy.dim() == 2:
            avg_entropy = entropy
        else:
            avg_entropy = entropy.unsqueeze(0) if entropy.dim() == 1 else entropy

        important_indices = []
        for batch_idx in range(attention_weights.size(0)):
            if batch_idx >= avg_entropy.size(0):
                important_indices.append(torch.tensor([]))
                continue

            batch_entropy = avg_entropy[batch_idx]
            mask = self._batch_mask(attention_mask, batch_idx)

            # Only consider non-padded positions
            if batch_entropy.dim() == 0:
                # If batch_entropy is scalar, create a tensor with same shape as mask
                batch_entropy = batch_entropy.repeat(mask.size(0))

            valid_entropy = batch_entropy[mask]
            if len(valid_entropy) == 0:
                important_indices.append(torch.tensor([]))
                continue

            valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
            selected_indices = self._select_top_p_then_elbow(
                valid_entropy,
                valid_indices,
                higher=True,
                context="entropy_elbow_higher"
            )
            if self.debug_enabled:
                total_tokens = int(mask.to(torch.int32).sum().item())
                candidates = len(valid_entropy)
                top_p_value = float(getattr(self.config, 'ATTENTION_TOP_P', 0.15) or 0.0)
                self._maybe_log_filter_summary(
                    category="entropy_elbow_higher_summary",
                    batch_idx=batch_idx,
                    total_tokens=total_tokens,
                    selected_tokens=len(selected_indices),
                    extra={
                        "candidates": candidates,
                        "top_p": f"{top_p_value:.4f}",
                    },
                )
            important_indices.append(selected_indices)
            
        return important_indices

    def max_attention_elbow_lower(self, attention_weights, attention_mask):
        """Filter tokens with high max attention using top-p + elbow method"""
        # Get maximum attention for each token
        max_attention = attention_weights.max(dim=-1)[0]  # [batch_size, seq_len]

        # Ensure max_attention has correct dimensions
        if max_attention.dim() > 2:
            while max_attention.dim() > 2:
                max_attention = max_attention.mean(dim=1)

        important_indices = []
        for batch_idx in range(attention_weights.size(0)):
            if batch_idx >= max_attention.size(0):
                important_indices.append(torch.tensor([]))
                continue

            batch_max_attn = max_attention[batch_idx]
            mask = self._batch_mask(attention_mask, batch_idx)

            # Handle scalar case
            if batch_max_attn.dim() == 0:
                batch_max_attn = batch_max_attn.repeat(mask.size(0))

            valid_max_attn = batch_max_attn[mask]
            if len(valid_max_attn) == 0:
                important_indices.append(torch.tensor([]))
                continue

            valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
            selected_indices = self._select_top_p_then_elbow(
                valid_max_attn,
                valid_indices,
                higher=True,
                context="max_attention_elbow_lower"
            )
            if self.debug_enabled:
                total_tokens = int(mask.to(torch.int32).sum().item())
                candidates = len(valid_max_attn)
                top_p_value = float(getattr(self.config, 'ATTENTION_TOP_P', 0.15) or 0.0)
                self._maybe_log_filter_summary(
                    category="max_attention_elbow_lower_summary",
                    batch_idx=batch_idx,
                    total_tokens=total_tokens,
                    selected_tokens=len(selected_indices),
                    extra={
                        "candidates": candidates,
                        "top_p": f"{top_p_value:.4f}",
                    },
                )
            important_indices.append(selected_indices)
            
        return important_indices

    def removed_avg_attention_elbow_higher(self, attention_weights, attention_mask):
        """Filter tokens with higher attention after removing average"""
        # Calculate token importance (sum of attention)
        token_importance = attention_weights.sum(dim=-1)  # [batch_size, seq_len]

        # Ensure token_importance has correct dimensions
        if token_importance.dim() > 2:
            while token_importance.dim() > 2:
                token_importance = token_importance.mean(dim=1)

        important_indices = []
        for batch_idx in range(attention_weights.size(0)):
            if batch_idx >= token_importance.size(0):
                important_indices.append(torch.tensor([]))
                continue

            batch_importance = token_importance[batch_idx]
            mask = self._batch_mask(attention_mask, batch_idx)

            # Handle scalar case
            if batch_importance.dim() == 0:
                batch_importance = batch_importance.repeat(mask.size(0))

            valid_importance = batch_importance[mask]
            if len(valid_importance) == 0:
                important_indices.append(torch.tensor([]))
                continue

            valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
            avg_importance = valid_importance.mean()
            relative_importance = valid_importance - avg_importance
            selected_indices = self._select_top_p_then_elbow(
                relative_importance,
                valid_indices,
                higher=True,
                context="removed_avg_elbow_higher"
            )
            if self.debug_enabled:
                total_tokens = int(mask.to(torch.int32).sum().item())
                candidates = len(relative_importance)
                top_p_value = float(getattr(self.config, 'ATTENTION_TOP_P', 0.15) or 0.0)
                self._maybe_log_filter_summary(
                    category="removed_avg_elbow_higher_summary",
                    batch_idx=batch_idx,
                    total_tokens=total_tokens,
                    selected_tokens=len(selected_indices),
                    extra={
                        "candidates": candidates,
                        "top_p": f"{top_p_value:.4f}",
                    },
                )
            important_indices.append(selected_indices)
            
        return important_indices
    
    def top_k_avg_attention_elbow_lower(self, attention_weights, attention_mask, top_k=None):
        """Filter tokens with lower average attention among top-k"""
        if top_k is None:
            top_k = self.config.ATTENTION_TOP_K

        # Get average attention for each token
        avg_attention = attention_weights.mean(dim=-1)  # [batch_size, seq_len]

        # Ensure avg_attention has correct dimensions
        if avg_attention.dim() > 2:
            while avg_attention.dim() > 2:
                avg_attention = avg_attention.mean(dim=1)

        important_indices = []
        for batch_idx in range(attention_weights.size(0)):
            if batch_idx >= avg_attention.size(0):
                important_indices.append(torch.tensor([]))
                continue

            batch_avg_attn = avg_attention[batch_idx]
            mask = self._batch_mask(attention_mask, batch_idx)

            # Handle scalar case
            if batch_avg_attn.dim() == 0:
                batch_avg_attn = batch_avg_attn.repeat(mask.size(0))
            
            valid_avg_attn = batch_avg_attn[mask]
            if len(valid_avg_attn) == 0:
                important_indices.append(torch.tensor([]))
                continue
            
            # Get top-k tokens by average attention
            actual_k = min(top_k, len(valid_avg_attn))
            top_k_values, top_k_indices = torch.topk(valid_avg_attn, actual_k)
            
            # Apply elbow method to top-k values
            elbow_idx = self.find_elbow_point(top_k_values, higher=False)
            threshold = torch.sort(top_k_values)[0][elbow_idx]
            
            # Filter top-k tokens with lower attention
            valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
            low_attn_mask = top_k_values <= threshold
            selected_indices = valid_indices[top_k_indices[low_attn_mask]]
            
            important_indices.append(selected_indices)
            
        return important_indices
    
    def sequential_filtering(self, attention_weights, attention_mask):
        """Apply multiple filtering methods sequentially"""
        methods = [
            self.max_attention_elbow_lower,
            self.removed_avg_attention_elbow_higher,
            self.top_k_avg_attention_elbow_lower,
            self.attention_entropy_elbow_higher,
        ]
        
        # Start with all valid positions
        batch_size = attention_weights.size(0)
        candidate_indices = []
        
        for batch_idx in range(batch_size):
            mask = self._batch_mask(attention_mask, batch_idx)
            candidates = torch.nonzero(mask, as_tuple=False).flatten()
            candidate_indices.append(candidates)
        
        # Apply each filtering method
        for method in methods:
            filtered_indices = method(attention_weights, attention_mask)
            
            # Intersect with previous candidates
            new_candidates = []
            for batch_idx in range(batch_size):
                if len(candidate_indices[batch_idx]) > 0 and len(filtered_indices[batch_idx]) > 0:
                    # Find intersection
                    intersection = torch.tensor([
                        idx.item() for idx in candidate_indices[batch_idx]
                        if idx.item() in filtered_indices[batch_idx].tolist()
                    ], dtype=torch.long)
                    new_candidates.append(intersection)
                else:
                    new_candidates.append(torch.tensor([]))
            
            candidate_indices = new_candidates
            
            # If no candidates left, break
            if all(len(candidates) == 0 for candidates in candidate_indices):
                break
        
        return candidate_indices
    
    def identify_important_tokens(self, attention_weights, attention_mask, method='basic', top_k=None):
        """
        Identify important tokens using various methods
        
        Args:
            attention_weights: [batch_size, seq_len, seq_len]
            attention_mask: [batch_size, seq_len]
            method: 'basic', 'entropy_elbow_higher', 'max_attention_elbow_lower', 
                   'removed_avg_elbow_higher', 'top_k_avg_elbow_lower', 'sequential'
            top_k: Number of tokens to identify (default: config.ATTENTION_TOP_K)
        
        Returns:
            important_token_indices: List of [batch_indices] for each batch
        """
        if method == 'entropy_elbow_higher':
            return self.attention_entropy_elbow_higher(attention_weights, attention_mask)
        elif method == 'max_attention_elbow_lower':
            return self.max_attention_elbow_lower(attention_weights, attention_mask)
        elif method == 'removed_avg_elbow_higher':
            return self.removed_avg_attention_elbow_higher(attention_weights, attention_mask)
        elif method == 'top_k_avg_elbow_lower':
            return self.top_k_avg_attention_elbow_lower(attention_weights, attention_mask, top_k)
        elif method == 'sequential':
            return self.sequential_filtering(attention_weights, attention_mask)
        else:  # basic method
            if top_k is None:
                top_k = self.config.ATTENTION_TOP_K
                
            batch_size, seq_len, _ = attention_weights.shape
            
            # Sum attention weights across all positions to get token importance
            token_importance = attention_weights.sum(dim=-1)
            
            # Mask padding tokens
            token_importance = token_importance * attention_mask.float()
            
            # Get top-k important tokens
            important_indices = []
            for batch_idx in range(batch_size):
                batch_importance = token_importance[batch_idx]
                mask = attention_mask[batch_idx].bool()
                
                valid_importance = batch_importance[mask]
                if len(valid_importance) == 0:
                    important_indices.append(torch.tensor([]))
                    continue
                
                actual_k = min(top_k, len(valid_importance))
                _, indices = torch.topk(valid_importance, actual_k)
                
                # Convert back to original sequence indices
                valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
                selected_indices = valid_indices[indices]
                important_indices.append(selected_indices)
            
            return important_indices
    
    def generate_hard_negatives(self, input_ids, attention_mask, model):
        """
        Generate hard negative samples by masking important tokens using advanced filtering
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            model: Trained model for attention extraction
            
        Returns:
            masked_input_ids: [batch_size, seq_len] - Input with important tokens masked
            original_tokens: List[List] - Original important tokens for each batch
        """
        # Extract attention weights
        attention_weights = self.extract_attention_weights(model, input_ids, attention_mask)
        
        # Identify important tokens using the configured filtering method
        important_indices = self.identify_important_tokens(
            attention_weights, 
            attention_mask, 
            method=self.config.ATTENTION_FILTERING_METHOD,
            top_k=self.config.ATTENTION_TOP_K
        )
        
        # Create masked copies
        masked_input_ids = input_ids.clone()
        original_tokens = []
        masked_positions_per_sample: List[set] = [set() for _ in range(input_ids.size(0))]

        for batch_idx in range(input_ids.size(0)):
            batch_original_tokens = []

            # Check if we have valid indices for this batch
            if len(important_indices[batch_idx]) > 0:
                for token_idx in important_indices[batch_idx]:
                    # Store original token
                    original_token = input_ids[batch_idx, token_idx].item()
                    batch_original_tokens.append(original_token)
                    
                    # Mask with probability
                    if torch.rand(1).item() < self.config.MASKING_PROBABILITY:
                        masked_input_ids[batch_idx, token_idx] = self.mask_token_id
                        masked_positions_per_sample[batch_idx].add(int(token_idx))
            else:
                # Fallback: if no important tokens found, use random masking
                valid_positions = torch.nonzero(attention_mask[batch_idx]).squeeze(-1)
                if len(valid_positions) > 0:
                    # Select a few random positions to mask
                    num_to_mask = min(self.config.ATTENTION_TOP_K, len(valid_positions))
                    random_indices = torch.randperm(len(valid_positions))[:num_to_mask]
                    
                    for idx in random_indices:
                        token_idx = valid_positions[idx]
                        original_token = input_ids[batch_idx, token_idx].item()
                        batch_original_tokens.append(original_token)
                        
                        if torch.rand(1).item() < self.config.MASKING_PROBABILITY:
                            masked_input_ids[batch_idx, token_idx] = self.mask_token_id
                            masked_positions_per_sample[batch_idx].add(int(token_idx))
                    
            original_tokens.append(batch_original_tokens)
        
        return masked_input_ids, original_tokens, masked_positions_per_sample
    
    def create_attention_guided_batch(self, batch, model):
        """
        Create a batch with both original samples and attention-guided hard negatives
        
        Args:
            batch: Original batch containing input_ids, attention_mask, labels
            model: Model for attention extraction
            
        Returns:
            combined_batch: Batch with original + hard negative samples
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch.get('label', batch.get('labels'))
        
        # Generate hard negatives
        masked_input_ids, _, masked_positions_per_sample = self.generate_hard_negatives(input_ids, attention_mask, model)

        # Filter out samples where no masking occurred
        keep_mask = torch.tensor([
            len(masked_positions_per_sample[idx]) > 0
            for idx in range(len(masked_positions_per_sample))
        ], dtype=torch.bool, device=input_ids.device)

        if keep_mask.any():
            filtered_masked_input_ids = masked_input_ids[keep_mask]
            filtered_attention_mask = attention_mask[keep_mask]
        else:
            filtered_masked_input_ids = torch.empty((0,) + masked_input_ids.shape[1:], dtype=masked_input_ids.dtype, device=masked_input_ids.device)
            filtered_attention_mask = torch.empty((0,) + attention_mask.shape[1:], dtype=attention_mask.dtype, device=attention_mask.device)

        # Combine original and masked samples
        combined_input_ids = torch.cat([input_ids, filtered_masked_input_ids], dim=0)
        combined_attention_mask = torch.cat([attention_mask, filtered_attention_mask], dim=0)

        # Create labels for hard negatives (mark as out-of-distribution)
        if labels is not None:
            hard_negative_labels = torch.full((filtered_masked_input_ids.size(0),), -1, dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([labels, hard_negative_labels], dim=0)
        else:
            combined_labels = None
            
        return {
            'input_ids': combined_input_ids,
            'attention_mask': combined_attention_mask,
            'label': combined_labels,
            'is_hard_negative': torch.cat([
                torch.zeros(input_ids.size(0), dtype=torch.bool),
                torch.ones(filtered_masked_input_ids.size(0), dtype=torch.bool)
            ], dim=0)
        }

# =============================================================================
# Dataset Loading
# =============================================================================

class NLPDatasetLoader:
    """Memory-efficient NLP dataset loader"""

    _dataset_cache = {}
    _max_cache_size = 2  # Limit to 2 datasets in memory

    @staticmethod
    def load_any_dataset(dataset_key: str, split: str = 'train') -> Optional[Dict[str, Any]]:
        """Load dataset by key with memory management"""
        if dataset_key not in Config.NLP_DATASETS:
            raise ValueError(f"Dataset key '{dataset_key}' not found in Config.NLP_DATASETS")

        cache_key = f"{dataset_key}_{split}"
        if cache_key in NLPDatasetLoader._dataset_cache:
            print(f"Using cached dataset: {dataset_key} (split: {split})")
            return NLPDatasetLoader._dataset_cache[cache_key]
        
        params = Config.NLP_DATASETS[dataset_key]
        
        try:
            if dataset_key == 'wikitext':
                data = NLPDatasetLoader._load_wikitext(split)
            else:
                data = NLPDatasetLoader._load_and_extract(
                    name=params['name'],
                    subset=params.get('subset'),
                    text_col=params['text_column'],
                    label_col=params.get('label_column'),
                    split=split,
                    trust_remote_code=params.get('trust_remote_code', False)
                )
            
            if data:
                # Memory management: clear old cache if too many datasets
                if len(NLPDatasetLoader._dataset_cache) >= NLPDatasetLoader._max_cache_size:
                    # Remove oldest cached dataset
                    oldest_key = next(iter(NLPDatasetLoader._dataset_cache))
                    del NLPDatasetLoader._dataset_cache[oldest_key]
                    print(f"Cleared cache for: {oldest_key}")

                NLPDatasetLoader._dataset_cache[cache_key] = data
                print(f"Dataset loaded and cached: {dataset_key} (split: {split})")
            else:
                print(f"Failed to load dataset: {dataset_key} (split: {split})")

            return data
            
        except Exception as e:
            print(f"Error loading dataset {dataset_key}: {e}")
            return None
    
    @staticmethod
    def _load_wikitext(split: str = 'train') -> Optional[Dict[str, Any]]:
        """Load WikiText dataset"""
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')[split]
            texts = [item['text'] for item in dataset if item['text'].strip()]
            # Filter out empty and very short texts
            texts = [text for text in texts if len(text.strip()) > 10]
            return {'text': texts, 'label': None}
        except Exception as e:
            print(f"Error loading WikiText: {e}")
            return None
    
    @staticmethod
    def _load_and_extract(name: str, subset: Optional[str], text_col: str, 
                         label_col: Optional[str], split: str, trust_remote_code: bool = False, 
                         fallback: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load and extract data from HuggingFace dataset with fallback support"""
        try:
            if subset:
                dataset = load_dataset(name, subset, trust_remote_code=trust_remote_code)[split]
            else:
                dataset = load_dataset(name, trust_remote_code=trust_remote_code)[split]
            
            texts = [item[text_col] for item in dataset]
            
            if label_col:
                labels = [item[label_col] for item in dataset]
                return {'text': texts, 'label': labels}
            else:
                return {'text': texts, 'label': None}
                
        except Exception as e:
            print(f"⚠️  Failed to load {name}: {e}")
            if fallback:
                print(f"🔄 Attempting fallback to {fallback}")
                try:
                    fallback_dataset = load_dataset(fallback, trust_remote_code=trust_remote_code)[split]
                    texts = [item[text_col] for item in fallback_dataset]
                    if label_col and label_col in fallback_dataset.column_names:
                        labels = [item[label_col] for item in fallback_dataset]
                        return {'text': texts, 'label': labels}
                    else:
                        return {'text': texts, 'label': None}
                except Exception as fe:
                    print(f"❌ Fallback also failed: {fe}")
                    return None
        except Exception as e:
            print(f"Error loading dataset {name}: {e}")
            return None
    
    @classmethod
    def clear_cache(cls):
        """Clear dataset cache"""
        cls._dataset_cache.clear()
        print("Dataset cache cleared.")

# =============================================================================
# PyTorch Dataset Classes  
# =============================================================================

class TextDataset(Dataset):
    """Simple text dataset for classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = preprocess_text_for_roberta(str(self.texts[idx]))
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class OSRTextDataset(Dataset):
    """Dataset for OSR training with OE support"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels  # -1 for OE samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = preprocess_text_for_roberta(str(self.texts[idx]))
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class CachedAttentionOEDataset(Dataset):
    """
    On-the-fly OE generation from Stage2 cached attention scores.

    This dataset generates OE samples dynamically during training by:
    1. Loading attention metadata from Stage2 cache
    2. Identifying important tokens based on global threshold
    3. Applying probabilistic masking to create hard negatives

    This enables Hendrycks-style simultaneous ID+OE training without
    pre-materializing all OE samples.
    """

    def __init__(
        self,
        cache_dir: Path,
        tokenizer,
        config,
        max_length: int = 256
    ):
        """
        Args:
            cache_dir: Path to Stage2 cache directory (contains metadata.json, stage2/*.pt)
            tokenizer: HuggingFace tokenizer
            config: Configuration object with ATTENTION_TOP_P, MASKING_PROBABILITY, etc.
            max_length: Maximum sequence length
        """
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length

        # Load metadata
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Stage2 metadata not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load Stage2 shards
        stage2_dir = self.cache_dir / "stage2"
        if not stage2_dir.exists():
            raise FileNotFoundError(f"Stage2 shard directory not found at {stage2_dir}")

        self.shard_files = sorted(stage2_dir.glob("shard_*.pt"))
        if not self.shard_files:
            raise FileNotFoundError(f"No Stage2 shards found in {stage2_dir}")

        # Build index: (shard_idx, local_idx) for each global sample
        self.sample_index = []
        self.total_samples = 0

        for shard_idx, shard_path in enumerate(self.shard_files):
            try:
                payload = torch.load(shard_path, map_location='cpu')
                records = payload.get('records', [])
                shard_size = len(records)

                for local_idx in range(shard_size):
                    self.sample_index.append((shard_idx, local_idx))
                    self.total_samples += 1
            except Exception as exc:
                print(f"⚠️  Warning: Failed to load shard {shard_path}: {exc}")
                continue

        if self.total_samples == 0:
            raise ValueError(f"No valid samples found in Stage2 cache at {cache_dir}")

        # Get configuration
        self.masking_probability = float(getattr(config, 'MASKING_PROBABILITY', 0.15))
        self.mask_token_id = tokenizer.mask_token_id

        print(f"✅ CachedAttentionOEDataset initialized: {self.total_samples} samples from {len(self.shard_files)} shards")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Generate an OE sample on-the-fly by:
        1. Loading the cached attention record
        2. Selecting high-attention tokens based on global threshold
        3. Probabilistically masking them
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")

        # Find shard and local index
        shard_idx, local_idx = self.sample_index[idx]

        # Load shard (cached in memory would be better, but this is simpler)
        shard_path = self.shard_files[shard_idx]
        payload = torch.load(shard_path, map_location='cpu')
        records = payload.get('records', [])
        record = records[local_idx]

        # Extract data from record
        input_ids = torch.tensor(record['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(record['attention_mask'], dtype=torch.long)
        scores = record.get('scores', [])
        indices = record.get('indices', [])
        spans = record.get('spans', [])

        # Create modified input_ids with masking
        modified_input_ids = input_ids.clone()

        # Apply masking based on cached important tokens
        # indices contains positions of high-attention tokens
        for span in spans:
            # Each span is a list of token positions forming a word
            if torch.rand(1).item() < self.masking_probability:
                for pos in span:
                    if pos < len(modified_input_ids) and attention_mask[pos]:
                        modified_input_ids[pos] = self.mask_token_id

        # Ensure proper length
        if len(modified_input_ids) > self.max_length:
            modified_input_ids = modified_input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        elif len(modified_input_ids) < self.max_length:
            padding_length = self.max_length - len(modified_input_ids)
            modified_input_ids = torch.cat([
                modified_input_ids,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])

        return {
            'input_ids': modified_input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(-1, dtype=torch.long)  # -1 for OE samples
        }

# =============================================================================
# Lightning Data Module
# =============================================================================

class SimpleDataModule(pl.LightningDataModule):
    """Simplified Lightning DataModule for OE experiments"""
    
    def __init__(self, config: Config, dataset_name: str):
        super().__init__()
        self.config = config
        self.dataset_name = dataset_name
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = 0
        self.label2id = {}
        self.id2label = {}
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        print(f"Setting up data for dataset: {self.dataset_name}")
        
        # Load training data
        train_data = NLPDatasetLoader.load_any_dataset(self.dataset_name, split='train')
        if not train_data or not train_data.get('text'):
            raise ValueError(f"Failed to load train data for {self.dataset_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            cache_dir=self.config.HUGGINGFACE_CACHE_DIR
        )
        
        # Create label mappings
        if train_data.get('label') is not None:
            unique_labels = sorted(set(train_data['label']))
            self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
            self.num_classes = len(unique_labels)
            
            # Convert labels to ids
            train_labels = [self.label2id[label] for label in train_data['label']]
        else:
            raise ValueError(f"Dataset {self.dataset_name} has no labels")
        
        # Split train into train/val (80/20)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_data['text'], train_labels, test_size=0.2, 
            random_state=self.config.RANDOM_STATE, stratify=train_labels
        )
        
        # Create datasets
        self.train_dataset = TextDataset(
            train_texts, train_labels, self.tokenizer, self.config.MAX_LENGTH
        )
        self.val_dataset = TextDataset(
            val_texts, val_labels, self.tokenizer, self.config.MAX_LENGTH
        )
        
        # Try to load test data
        try:
            test_data = NLPDatasetLoader.load_any_dataset(self.dataset_name, split='test')
            if test_data and test_data.get('text') and test_data.get('label'):
                test_labels = [self.label2id[label] for label in test_data['label'] 
                             if label in self.label2id]
                test_texts = [test_data['text'][i] for i, label in enumerate(test_data['label'])
                             if label in self.label2id]
                self.test_dataset = TextDataset(
                    test_texts, test_labels, self.tokenizer, self.config.MAX_LENGTH
                )
                print(f"Test dataset loaded: {len(test_texts)} samples")
            else:
                print("Using validation set as test set")
                self.test_dataset = self.val_dataset
        except:
            print("Using validation set as test set") 
            self.test_dataset = self.val_dataset
        
        print(f"Dataset setup complete:")
        print(f"  - Train: {len(self.train_dataset)} samples")
        print(f"  - Val: {len(self.val_dataset)} samples") 
        print(f"  - Test: {len(self.test_dataset)} samples")
        print(f"  - Classes: {self.num_classes}")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )

# =============================================================================
# Lightning Model
# =============================================================================

class SimpleOEModel(pl.LightningModule):
    """Simplified OE Lightning Module following original GitHub implementation"""
    
    def __init__(self, config: Config, num_labels: int, label2id: dict, id2label: dict,
                 class_weights: Optional[torch.Tensor] = None, tokenizer=None,
                 use_self_attention_oe: bool = False, oe_dataset_provided: bool = False):
        super().__init__()
        self.config_params = config
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.use_self_attention_oe = use_self_attention_oe
        self.oe_dataset_provided = oe_dataset_provided

        self.save_hyperparameters(ignore=['config', 'config_params', 'class_weights', 'tokenizer'])

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
            cache_dir=config.HUGGINGFACE_CACHE_DIR
        )

        # Initialize Self Attention-guided OE if enabled for this specific scenario
        if self.use_self_attention_oe and not self.oe_dataset_provided and tokenizer is not None:
            self.attention_analyzer = SelfAttentionAnalyzer(config, tokenizer)
            print(f"✅ Self Attention-guided OE enabled with {config.ATTENTION_TOP_K} top-k tokens (on-the-fly)")
        else:
            self.attention_analyzer = None
            if self.use_self_attention_oe and self.oe_dataset_provided:
                print("✅ Self Attention-guided OE using pre-generated dataset")
            else:
                print("⚠️  Self Attention-guided OE disabled (using traditional OE)")
        
        # Loss function
        if self.config_params.USE_WEIGHTED_LOSS and self.class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            print(f"Using weighted CrossEntropyLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print(f"Using standard CrossEntropyLoss")
        
        # Metrics
        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels, average='micro'),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='weighted'),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
    
    def setup(self, stage=None):
        if self.config_params.USE_WEIGHTED_LOSS and hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)
            print(f"Moved class weights to {self.device}")
    
    def forward(self, batch=None, input_ids=None, attention_mask=None, output_features=False, output_attentions=False):
        # Support both batch dict and direct keyword arguments
        if batch is not None:
            input_ids = batch.get('input_ids')
            attention_mask = batch.get('attention_mask')
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_features,
            output_attentions=output_attentions
        )
    
    def _common_step(self, batch, batch_idx):
        oe_batch = None

        # Handle combined OE+ID batch format first
        if isinstance(batch, tuple) and len(batch) == 2:
            # Check if this is (id_batch, oe_batch) from CombinedOSRDataLoader
            first_item = batch[0]
            if isinstance(first_item, dict) and 'input_ids' in first_item:
                # This is (id_batch, oe_batch) format
                id_batch, oe_batch = batch
                batch = id_batch

        # Handle dict and tuple batch formats
        if isinstance(batch, tuple):
            if len(batch) == 3:
                input_ids, attention_mask, labels = batch
            elif len(batch) == 2:
                input_ids, labels = batch
                attention_mask = None
            else:
                raise ValueError(f"Unexpected batch tuple length: {len(batch)}")
        else:
            # Dict format
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

        # Ensure all inputs are tensors - check for dict/list first
        if isinstance(input_ids, dict):
            raise ValueError(f"input_ids is still a dict: {input_ids}")
        elif isinstance(input_ids, list):
            input_ids = torch.stack(input_ids)
        elif not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=self.device)

        if attention_mask is not None:
            if isinstance(attention_mask, dict):
                raise ValueError(f"attention_mask is still a dict: {attention_mask}")
            elif isinstance(attention_mask, list):
                attention_mask = torch.stack(attention_mask)
            elif not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask, device=self.device)

        if isinstance(labels, dict):
            raise ValueError(f"labels is still a dict: {labels}")
        elif isinstance(labels, list):
            labels = torch.tensor(labels, device=self.device)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=self.device)
        
        # Check if we need to generate Self Attention-guided hard negatives
        if (self.training and
            self.use_self_attention_oe and
            not self.oe_dataset_provided and
            hasattr(self, 'attention_analyzer') and
            self.attention_analyzer is not None):

            # Convert tuple batch to dict for attention analyzer
            batch_dict = {}
            if isinstance(batch, tuple):
                if len(batch) == 3:
                    batch_dict['input_ids'] = batch[0]
                    batch_dict['attention_mask'] = batch[1]
                    batch_dict['label'] = batch[2]
                elif len(batch) == 2:
                    batch_dict['input_ids'] = batch[0]
                    batch_dict['label'] = batch[1]
                    batch_dict['attention_mask'] = None
            else:
                batch_dict = batch

            # Generate attention-guided batch with hard negatives
            enhanced_batch = self.attention_analyzer.create_attention_guided_batch(batch_dict, self.model)
            
            # Use enhanced batch for training
            input_ids = enhanced_batch['input_ids'] 
            attention_mask = enhanced_batch['attention_mask']
            labels = enhanced_batch['label']
            
            # Handle hard negative labels (convert -1 to a valid class for loss computation)
            if labels is not None:
                # Replace -1 (hard negatives) with a valid class index for loss computation
                # We'll compute loss separately for ID and hard negatives
                id_mask = enhanced_batch['is_hard_negative'] == False
                hard_neg_mask = enhanced_batch['is_hard_negative'] == True
                
                if id_mask.sum() > 0 and hard_neg_mask.sum() > 0:
                    # Compute ID loss (normal classification)
                    id_outputs = self.model(
                        input_ids=input_ids[id_mask],
                        attention_mask=attention_mask[id_mask]
                    )
                    id_loss = F.cross_entropy(id_outputs.logits, labels[id_mask])
                    
                    # Compute OE loss for hard negatives (entropy maximization)
                    oe_outputs = self.model(
                        input_ids=input_ids[hard_neg_mask],
                        attention_mask=attention_mask[hard_neg_mask] 
                    )
                    # OE loss: maximize entropy (minimize confidence)
                    oe_logits = oe_outputs.logits
                    oe_loss = -(F.log_softmax(oe_logits, dim=1).mean())
                    
                    # Combined loss
                    loss = id_loss + float(getattr(self.config_params, 'SELF_ATTENTION_LOSS_WEIGHT', 0.5)) * oe_loss
                    
                    # For predictions, only use ID samples
                    logits = id_outputs.logits
                    preds = torch.argmax(logits, dim=1)
                    return loss, logits, preds, labels[id_mask]
                else:
                    # Fallback to regular processing if batch composition is unexpected
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)
                    return loss, logits, preds, labels
            
        # Regular processing (baseline or traditional OE)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        # Add OE loss if OE batch is available
        if oe_batch is not None and self.training:
            try:
                oe_input_ids = oe_batch['input_ids']
                oe_attention_mask = oe_batch.get('attention_mask')

                if isinstance(oe_input_ids, list):
                    oe_input_ids = torch.stack(oe_input_ids)
                if not isinstance(oe_input_ids, torch.Tensor):
                    oe_input_ids = torch.tensor(oe_input_ids, dtype=torch.long)

                if oe_attention_mask is not None:
                    if isinstance(oe_attention_mask, list):
                        oe_attention_mask = torch.stack(oe_attention_mask)
                    if not isinstance(oe_attention_mask, torch.Tensor):
                        oe_attention_mask = torch.tensor(oe_attention_mask, dtype=torch.long)

                oe_input_ids = oe_input_ids.to(self.device)
                if oe_attention_mask is not None:
                    oe_attention_mask = oe_attention_mask.to(self.device)

                oe_outputs = self.model(
                    input_ids=oe_input_ids,
                    attention_mask=oe_attention_mask
                )
                oe_logits = oe_outputs.logits

                centered_logits = oe_logits - oe_logits.max(dim=1, keepdim=True)[0]
                uniform_log_probs = F.log_softmax(centered_logits, dim=1)
                oe_loss = -uniform_log_probs.mean()

                loss = loss + float(getattr(self.config_params, 'OE_UNIFORM_LOSS_WEIGHT', 1.0)) * oe_loss

            except Exception as e:
                print(f"Warning: OE processing failed: {e}")

        return loss, logits, preds, labels

    def training_step(self, batch, batch_idx):
        loss, logits, preds, labels = self._common_step(batch, batch_idx)
        self.train_metrics.update(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # Store predictions and labels for AUROC calculation
        if not hasattr(self, 'train_logits'):
            self.train_logits = []
            self.train_labels = []

        # Store every 10th batch to avoid memory issues
        if batch_idx % 10 == 0:
            self.train_logits.append(logits.detach().cpu())
            self.train_labels.extend(labels.cpu())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, preds, labels = self._common_step(batch, batch_idx)
        self.val_metrics.update(preds, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        # Store predictions and labels for AUROC calculation
        if not hasattr(self, 'val_logits'):
            self.val_logits = []
            self.val_labels = []

        self.val_logits.append(logits.detach().cpu())
        self.val_labels.extend(labels.cpu())

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss, _, preds, labels = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return {'test_loss': loss}
    
    def on_train_epoch_end(self):
        try:
            computed_metrics = self.train_metrics.compute()
            self.log_dict(computed_metrics, prog_bar=True)

            # Calculate and log AUROC if we have predictions
            if hasattr(self, 'train_logits') and len(self.train_logits) > 0:
                try:
                    import torch
                    from sklearn.metrics import roc_auc_score

                    train_logits = torch.cat(self.train_logits)
                    train_labels = torch.stack(self.train_labels)

                    if self.num_labels == 2:
                        train_scores = torch.softmax(train_logits, dim=1)[:, 1].cpu().numpy()
                        train_labels_np = train_labels.cpu().numpy().reshape(-1)

                        if len(set(train_labels_np)) > 1:
                            train_auroc = roc_auc_score(train_labels_np, train_scores)
                            self.log('train_auroc', train_auroc, prog_bar=True)

                            if hasattr(self.logger, 'experiment'):
                                self.logger.experiment.log({
                                    'train_auroc': train_auroc,
                                    'epoch': self.current_epoch
                                })
                    else:
                        train_probs = torch.softmax(train_logits, dim=1)
                        train_labels_tensor = train_labels

                        if len(torch.unique(train_labels_tensor)) > 1:
                            train_auroc = tmf.auroc(
                                train_probs,
                                train_labels_tensor,
                                task='multiclass',
                                num_classes=self.num_labels,
                                average='macro'
                            ).item()
                            self.log('train_auroc', train_auroc, prog_bar=True)

                            if hasattr(self.logger, 'experiment'):
                                self.logger.experiment.log({
                                    'train_auroc': train_auroc,
                                    'epoch': self.current_epoch
                                })

                    # Clear stored logits/labels
                    self.train_logits = []
                    self.train_labels = []

                except Exception as auroc_error:
                    print(f"Warning: Train AUROC calculation error: {auroc_error}")

            self.train_metrics.reset()
        except Exception as e:
            print(f"Warning: Train metrics error: {e}")
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        try:
            computed_metrics = self.val_metrics.compute()
            self.log_dict(computed_metrics, prog_bar=True)

            # Calculate and log AUROC if we have predictions
            if hasattr(self, 'val_logits') and len(self.val_logits) > 0:
                try:
                    import torch
                    from sklearn.metrics import roc_auc_score

                    val_logits = torch.cat(self.val_logits)
                    val_labels = torch.stack(self.val_labels)

                    if self.num_labels == 2:
                        val_scores = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
                        val_labels_np = val_labels.cpu().numpy().reshape(-1)

                        if len(set(val_labels_np)) > 1:
                            val_auroc = roc_auc_score(val_labels_np, val_scores)
                            self.log('val_auroc', val_auroc, prog_bar=True)

                            if hasattr(self.logger, 'experiment'):
                                self.logger.experiment.log({
                                    'val_auroc': val_auroc,
                                    'epoch': self.current_epoch
                                })
                    else:
                        val_probs = torch.softmax(val_logits, dim=1)
                        val_labels_tensor = val_labels

                        if len(torch.unique(val_labels_tensor)) > 1:
                            val_auroc = tmf.auroc(
                                val_probs,
                                val_labels_tensor,
                                task='multiclass',
                                num_classes=self.num_labels,
                                average='macro'
                            ).item()
                            self.log('val_auroc', val_auroc, prog_bar=True)

                            if hasattr(self.logger, 'experiment'):
                                self.logger.experiment.log({
                                    'val_auroc': val_auroc,
                                    'epoch': self.current_epoch
                                })

                    # Clear stored logits/labels
                    self.val_logits = []
                    self.val_labels = []

                except Exception as auroc_error:
                    print(f"Warning: AUROC calculation error: {auroc_error}")

            self.val_metrics.reset()
        except Exception as e:
            print(f"Warning: Validation metrics error: {e}")
            self.val_metrics.reset()
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config_params.LEARNING_RATE)
        
        # Calculate total steps for scheduler
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            num_training_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback calculation
            num_training_steps = len(self.train_dataloader()) * self.config_params.NUM_EPOCHS
        
        num_warmup_steps = int(num_training_steps * 0.1)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

# =============================================================================
# OSR Model with Original OE Loss
# =============================================================================

class OSRLightningModule(pl.LightningModule):
    """OSR Lightning Module with original GitHub-style OE loss"""
    
    def __init__(self, model_name: str, num_labels: int, learning_rate: float, 
                 cache_dir: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels, 
            cache_dir=cache_dir
        )
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)

        # Additional comprehensive metrics
        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average="macro")
        self.val_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average="micro")
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_labels, average="macro")
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_labels, average="macro")

        # Storage for epoch-end comprehensive evaluation
        self.validation_predictions = []
        self.validation_targets = []
        self.validation_logits = []
    
    def forward(self, input_ids, attention_mask, output_features=False):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if output_features:
            features = outputs.hidden_states[-1][:, 0, :]
            return logits, features
        return logits
    
    def training_step(self, batch, batch_idx):
        """
        Training step with original GitHub OE loss implementation:
        oe_loss = -1 * F.log_softmax(logits_oe - torch.max(logits_oe, dim=1, keepdim=True)[0], dim=1).mean()
        """
        # Handle combined batch (ID, OE) or single batch (ID only)
        if isinstance(batch, dict):
            # Single batch (ID only)
            id_batch = batch
            oe_batch = None
        else:
            # Combined batch (ID, OE)
            id_batch, oe_batch = batch
        
        # ID loss (standard cross-entropy)
        id_logits = self.model(id_batch['input_ids'], id_batch['attention_mask']).logits
        id_loss = F.cross_entropy(id_logits, id_batch['label'])
        
        # ID accuracy
        preds = torch.argmax(id_logits, dim=1)
        self.train_accuracy.update(preds, id_batch['label'])
        
        # OE loss (original GitHub implementation)
        if oe_batch is not None:
            oe_logits = self.model(oe_batch['input_ids'], oe_batch['attention_mask']).logits
            # Original GitHub OE loss: negative mean of log-softmax 
            smax_oe = F.log_softmax(
                oe_logits - torch.max(oe_logits, dim=1, keepdim=True)[0], 
                dim=1
            )
            oe_loss = -1 * smax_oe.mean()
        else:
            oe_loss = torch.tensor(0.0).to(self.device)
        
        # Combined loss (equal weighting as in original)
        total_loss = id_loss + oe_loss
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_id_loss', id_loss, on_step=False, on_epoch=True)
        self.log('train_oe_loss', oe_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.model(batch['input_ids'], batch['attention_mask']).logits
        loss = F.cross_entropy(logits, batch['label'])

        # Predictions
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_accuracy.update(preds, batch['label'])
        self.val_f1_macro.update(preds, batch['label'])
        self.val_f1_micro.update(preds, batch['label'])
        self.val_precision.update(preds, batch['label'])
        self.val_recall.update(preds, batch['label'])

        # Store for comprehensive evaluation at epoch end
        self.validation_predictions.extend(preds.cpu().tolist())
        self.validation_targets.extend(batch['label'].cpu().tolist())
        self.validation_logits.extend(logits.detach().cpu().tolist())

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        logits = self.model(batch['input_ids'], batch['attention_mask']).logits
        loss = F.cross_entropy(logits, batch['label'])
        
        preds = torch.argmax(logits, dim=1)
        
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return {'test_loss': loss}
    
    def on_train_epoch_end(self):
        self.log('train_acc', self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()
    
    def on_validation_epoch_end(self):
        # Compute and log torchmetrics
        val_acc = self.val_accuracy.compute()
        val_f1_macro = self.val_f1_macro.compute()
        val_f1_micro = self.val_f1_micro.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()

        # Log main metrics
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_f1_macro', val_f1_macro, prog_bar=True)
        self.log('val_f1_micro', val_f1_micro, prog_bar=True)
        self.log('val_precision', val_precision, prog_bar=True)
        self.log('val_recall', val_recall, prog_bar=True)

        # Calculate comprehensive metrics if we have collected data
        if self.validation_predictions and self.validation_targets:
            y_true = np.array(self.validation_targets)
            y_pred = np.array(self.validation_predictions)
            y_logits = np.array(self.validation_logits)

            # Get probabilities for binary classification AUROC
            y_scores = None
            if self.num_labels == 2:
                y_probs = torch.softmax(torch.tensor(y_logits), dim=1)
                y_scores = y_probs[:, 1].numpy()  # Probability of positive class

            # Calculate comprehensive metrics
            comprehensive_metrics = calculate_comprehensive_metrics(y_true, y_pred, y_scores)

            # Log comprehensive metrics with unique prefix to avoid conflicts
            for metric_name, metric_value in comprehensive_metrics.items():
                # Use 'sklearn_' prefix to avoid any conflicts with torchmetrics
                self.log(f'val_sklearn_{metric_name}', float(metric_value), prog_bar=False)

            # Print comprehensive epoch summary
            print(f"\n{'='*60}")
            print(f"EPOCH {self.current_epoch} VALIDATION METRICS SUMMARY")
            print(f"{'='*60}")
            print(f"Accuracy:           {val_acc:.4f}")
            print(f"F1-Score (Macro):   {val_f1_macro:.4f}")
            print(f"F1-Score (Micro):   {val_f1_micro:.4f}")
            print(f"Precision (Macro):  {val_precision:.4f}")
            print(f"Recall (Macro):     {val_recall:.4f}")

            if 'auroc' in comprehensive_metrics:
                print(f"AUROC:              {comprehensive_metrics['auroc']:.4f}")
                print(f"AUPR:               {comprehensive_metrics['aupr']:.4f}")

            # Per-class F1 scores
            if self.num_labels <= 10:
                f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
                print("\nPer-class F1 Scores:")
                for i, f1_val in enumerate(f1_per_class):
                    print(f"  Class {i}: {f1_val:.4f}")

            print(f"{'='*60}\n")

        # Reset metrics and storage
        self.val_accuracy.reset()
        self.val_f1_macro.reset()
        self.val_f1_micro.reset()
        self.val_precision.reset()
        self.val_recall.reset()

        # Clear stored data
        self.validation_predictions = []
        self.validation_targets = []
        self.validation_logits = []
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

# =============================================================================
# Combined DataLoader for OSR Training
# =============================================================================

class CombinedOSRDataLoader:
    """Yield (ID batch, OE batch) pairs with a fixed ID:OE sample ratio."""

    def __init__(self, id_loader, oe_loader=None, ratio: int = 1):
        self.id_loader = id_loader
        self.oe_loader = oe_loader
        self.dataset = getattr(id_loader, 'dataset', None)
        self.batch_size = getattr(id_loader, 'batch_size', None)
        self.ratio = max(1, int(ratio))

    def __len__(self):
        return len(self.id_loader)

    @staticmethod
    def _batch_size(batch) -> int:
        if isinstance(batch, dict):
            for key in ('input_ids', 'label', 'attention_mask'):
                if key in batch:
                    tensor = batch[key]
                    if isinstance(tensor, torch.Tensor):
                        return tensor.size(0)
                    return len(tensor)
            raise ValueError("Batch dictionary lacks tensor-like entries")
        if isinstance(batch, (list, tuple)):
            first = batch[0]
            if isinstance(first, torch.Tensor):
                return first.size(0)
            return len(first)
        raise ValueError("Unsupported batch format")

    @staticmethod
    def _split_batch(batch, count: int):
        if isinstance(batch, dict):
            taken = {}
            remaining = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    taken[key] = value[:count]
                    remaining[key] = value[count:]
                else:
                    taken[key] = value[:count]
                    remaining[key] = value[count:]
            return taken, remaining
        if isinstance(batch, (list, tuple)):
            taken = []
            remaining = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    taken.append(value[:count])
                    remaining.append(value[count:])
                else:
                    taken.append(value[:count])
                    remaining.append(value[count:])
            if isinstance(batch, tuple):
                taken = tuple(taken)
                remaining = tuple(remaining)
            return taken, remaining
        raise ValueError("Unsupported batch format in split")

    @staticmethod
    def _concat_batches(batches):
        if not batches:
            raise ValueError("No batches to concatenate")
        first = batches[0]
        if isinstance(first, dict):
            concatenated = {}
            for key in first.keys():
                values = [b[key] for b in batches]
                if isinstance(values[0], torch.Tensor):
                    concatenated[key] = torch.cat(values, dim=0)
                else:
                    concatenated[key] = sum(values, [])
            return concatenated
        if isinstance(first, (list, tuple)):
            stacked = []
            for idx in range(len(first)):
                values = [b[idx] for b in batches]
                if isinstance(values[0], torch.Tensor):
                    stacked.append(torch.cat(values, dim=0))
                else:
                    stacked.append(sum(values, []))
            return tuple(stacked) if isinstance(first, tuple) else stacked
        raise ValueError("Unsupported batch format in concat")

    def __iter__(self):
        id_iter = iter(self.id_loader)
        oe_iter = iter(self.oe_loader) if self.oe_loader else None
        oe_buffer = None

        for id_batch in id_iter:
            if oe_iter is None:
                yield id_batch
                continue

            required = max(1, math.ceil(self._batch_size(id_batch) / self.ratio))
            collected = []
            while required > 0:
                if oe_buffer is None or self._batch_size(oe_buffer) == 0:
                    try:
                        oe_buffer = next(oe_iter)
                    except StopIteration:
                        oe_iter = iter(self.oe_loader)
                        oe_buffer = next(oe_iter)

                available = self._batch_size(oe_buffer)
                take = min(required, available)
                taken, oe_buffer = self._split_batch(oe_buffer, take)
                collected.append(taken)
                required -= take

            oe_batch = self._concat_batches(collected)
            yield id_batch, oe_batch

# =============================================================================
# MSP-based OOD Evaluation (Original GitHub Implementation)
# =============================================================================

def calculate_msp_scores(model, dataloader, device):
    """
    Calculate MSP (Maximum Softmax Probability) scores.

    MSP represents the model's confidence in its prediction. Higher MSP indicates
    the model is more confident, which typically corresponds to in-distribution (ID) samples.
    Lower MSP indicates uncertainty, suggesting out-of-distribution (OOD) samples.

    Returns:
        np.ndarray: MSP scores in range [0, 1]
            - High MSP (→ 1.0) = Model is confident → Likely ID
            - Low MSP (→ 0.0) = Model is uncertain → Likely OOD

    Note:
        This function returns POSITIVE MSP values (confidence scores).
        For OOD detection metrics, these will be converted to OOD scores
        using (1 - MSP) in get_ood_metrics().
    """
    model.eval()
    all_scores = []

    # Check if dataloader is empty
    if not dataloader or len(dataloader.dataset) == 0:
        print(f"Warning: Empty dataloader provided to calculate_msp_scores")
        return np.array([])

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Handle both HuggingFace and Lightning model outputs
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # MSP calculation with numerical stability
            # Subtract max value before softmax to prevent overflow
            normalized_logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
            smax = F.softmax(normalized_logits, dim=1)

            # FIXED: Return positive MSP (confidence score)
            # Previously: msp = -1 * torch.max(smax, dim=1)[0]  # Bug: negative MSP
            msp = torch.max(smax, dim=1)[0]  # Correct: positive confidence

            all_scores.extend(msp.cpu().numpy())

    return np.array(all_scores)

def calculate_comprehensive_metrics(y_true, y_pred, y_scores=None):
    """Calculate comprehensive classification metrics"""
    metrics = {}

    # Basic classification metrics
    # Skip basic metrics as they're handled by TorchMetrics: accuracy, f1_macro, f1_micro, precision_macro, recall_macro
    # Only compute additional metrics not covered by TorchMetrics

    # Per-class metrics
    if len(np.unique(y_true)) <= 10:  # Only for reasonable number of classes
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, f1_val in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1_val

    # ROC AUC if scores are provided and binary classification
    if y_scores is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_scores)
            metrics['aupr'] = average_precision_score(y_true, y_scores)
        except ValueError:
            pass  # Skip if not possible

    return metrics

def get_ood_metrics(id_scores, ood_scores):
    """
    Calculate OOD detection metrics following standard OOD detection convention.

    Args:
        id_scores: MSP (confidence) scores for ID samples (from calculate_msp_scores)
        ood_scores: MSP (confidence) scores for OOD samples (from calculate_msp_scores)

    Returns:
        dict: OOD detection metrics
            - AUROC: Area Under ROC Curve (higher is better)
            - AUPR: Average Precision (higher is better)
            - FPR95: False Positive Rate at 95% True Positive Rate (lower is better)

    Convention (STANDARD - following Hendrycks et al. and OOD literature):
        - OOD samples are the POSITIVE class (label=1)
        - ID samples are the NEGATIVE class (label=0)
        - OOD score = 1 - MSP (higher OOD score indicates more likely OOD)
        - AUROC measures how well OOD scores separate OOD from ID
        - FPR95: "ID false positive rate when OOD recall (TPR) is 95%"

    Note:
        Input scores are MSP (confidence), which are converted to OOD scores
        by taking (1 - MSP). This ensures:
        - High MSP (confident) → Low OOD score → Likely ID
        - Low MSP (uncertain) → High OOD score → Likely OOD
    """
    # Check for empty inputs
    if len(id_scores) == 0 or len(ood_scores) == 0:
        print(f"Warning: Empty score arrays. ID: {len(id_scores)}, OOD: {len(ood_scores)}")
        return {'AUROC': np.nan, 'AUPR': np.nan, 'FPR95': np.nan}

    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)

    # Convert MSP (confidence) to OOD scores
    # High MSP (confident, ~1.0) → Low OOD score (~0.0) → Likely ID
    # Low MSP (uncertain, ~0.0) → High OOD score (~1.0) → Likely OOD
    id_ood_scores = 1.0 - id_scores
    ood_ood_scores = 1.0 - ood_scores

    # Combine scores and create labels (STANDARD CONVENTION)
    all_scores = np.concatenate([id_ood_scores, ood_ood_scores])
    all_labels = np.concatenate([
        np.zeros(len(id_ood_scores)),   # ID samples = 0 (negative class)
        np.ones(len(ood_ood_scores))    # OOD samples = 1 (positive class)
    ])

    # Check for invalid values
    if np.any(np.isnan(all_scores)) or np.any(np.isinf(all_scores)):
        print(f"Warning: Found NaN or Inf in scores. ID: {len(id_scores)}, OOD: {len(ood_scores)}")
        print(f"ID MSP range: [{np.nanmin(id_scores):.4f}, {np.nanmax(id_scores):.4f}]")
        print(f"OOD MSP range: [{np.nanmin(ood_scores):.4f}, {np.nanmax(ood_scores):.4f}]")
        print(f"ID OOD score range: [{np.nanmin(id_ood_scores):.4f}, {np.nanmax(id_ood_scores):.4f}]")
        print(f"OOD OOD score range: [{np.nanmin(ood_ood_scores):.4f}, {np.nanmax(ood_ood_scores):.4f}]")
        return {'AUROC': np.nan, 'AUPR': np.nan, 'FPR95': np.nan}

    # Calculate metrics with error handling
    try:
        auroc = roc_auc_score(all_labels, all_scores)
        aupr = average_precision_score(all_labels, all_scores)
    except ValueError as e:
        print(f"Error calculating AUROC/AUPR: {e}")
        return {'AUROC': np.nan, 'AUPR': np.nan, 'FPR95': np.nan}

    # FPR at 95% TPR
    thresholds = np.percentile(all_scores, np.linspace(0, 100, 1000))
    fpr_list = []

    for threshold in thresholds:
        predictions = (all_scores >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        tn = np.sum((predictions == 0) & (all_labels == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        if tpr >= 0.95:
            fpr_list.append(fpr)

    fpr95 = min(fpr_list) if fpr_list else 1.0

    return {
        'AUROC': auroc,
        'AUPR': aupr,
        'FPR95': fpr95
    }

def setup_pytorch_logger(log_dir: str, name: str, project: str = "simplified-oe", config_dict: dict = None, tags: list = None):
    """Setup PyTorch Lightning logger with fallback options"""

    # Get WandB API key from environment or config
    import os
    from datetime import datetime
    wandb_api_key = os.getenv('WANDB_API_KEY') or getattr(Config, 'WANDB_API_KEY', None)

    if WANDB_AVAILABLE and wandb_api_key:
        try:
            # Create unique run name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_name = f"{name}_{timestamp}"

            print(f"🔗 Connecting to WandB project: bang001-ku/{project}")
            print(f"📝 Run name: {unique_name}")

            # Setup config for WandB
            wandb_config = {}
            if config_dict:
                wandb_config.update(config_dict)

            # Add experiment metadata
            import sys
            wandb_config.update({
                'timestamp': timestamp,
                'run_name': unique_name,
                'log_dir': log_dir,
                'execution_command': ' '.join(sys.argv),
                'working_directory': os.getcwd()
            })

            return WandbLogger(
                entity="bang001-ku",
                project=project,
                name=unique_name,
                save_dir=log_dir,
                offline=False,
                config=wandb_config,
                tags=tags or []
            )
        except Exception as e:
            print(f"⚠️  WandB setup failed: {e}, using default logging")
    else:
        print("⚠️  WandB API key not found, using default logging")
        print("💡 To use WandB: export WANDB_API_KEY=your_key or run 'wandb login'")

    # Return None to use PyTorch Lightning's default logger
    return None

# =============================================================================
# Main Pipeline
# =============================================================================

class SimplifiedOEPipeline:
    """Simplified OE pipeline following original GitHub approach"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_module = None
        # Get logger for current dataset
        self.logger = logging.getLogger(f"oe_experiment_{config.CURRENT_NLP_DATASET}")

    # ------------------------------------------------------------------
    # Attention cache helpers (staged mode)
    # ------------------------------------------------------------------

    def _resolve_attention_cache_paths(self, algorithm: str) -> Dict[str, Path]:
        dataset_name = self.config.CURRENT_NLP_DATASET
        base_dir = Path(self.config.ATTENTION_CACHE_DIR)

        # Use centralized cache path resolution
        variant_dir = resolve_attention_cache_path(
            base_dir=base_dir,
            dataset=dataset_name,
            algorithm=algorithm,
            top_p=getattr(self.config, 'ATTENTION_TOP_P', None)
        )

        stage2_dir = variant_dir / "stage2"
        metadata_path = variant_dir / "metadata.json"
        manifest_path = variant_dir / "stage2_manifest.json"
        return {
            "run_dir": variant_dir,
            "stage2_dir": stage2_dir,
            "metadata": metadata_path,
            "manifest": manifest_path,
        }

    def _initialise_stage2_cache(self, paths: Dict[str, Path]) -> None:
        stage2_dir = paths["stage2_dir"]
        if stage2_dir.exists():
            for child in stage2_dir.glob("*.pt"):
                try:
                    child.unlink()
                except OSError:
                    pass
        else:
            stage2_dir.mkdir(parents=True, exist_ok=True)

    def _write_stage2_shard(
        self,
        shard_records: List[Dict[str, Any]],
        shard_index: int,
        paths: Dict[str, Path]
    ) -> Path:
        shard_name = f"shard_{shard_index:05d}.pt"
        shard_path = paths["stage2_dir"] / shard_name
        payload = {
            "records": shard_records,
            "shard_index": shard_index,
        }
        torch.save(payload, shard_path)
        return shard_path

    def _load_stage2_candidates(self, paths: Dict[str, Path]) -> List[Dict[str, Any]]:
        stage2_dir = paths["stage2_dir"]
        if not stage2_dir.exists():
            raise FileNotFoundError(
                f"Stage2 cache not found at {stage2_dir}. Run with --attention_stage stage2 first."
            )
        all_records: List[Dict[str, Any]] = []
        shard_paths = sorted(stage2_dir.glob("shard_*.pt"))
        for shard_path in shard_paths:
            try:
                payload = torch.load(shard_path, map_location="cpu")
            except Exception as exc:
                print(f"⚠️  Failed to load shard {shard_path}: {exc}")
                continue
            shard_records = payload.get("records", [])
            if isinstance(shard_records, list):
                all_records.extend(shard_records)
        return all_records

    def _write_stage2_metadata(
        self,
        paths: Dict[str, Path],
        *,
        config_snapshot: Dict[str, Any],
        total_samples: int,
        processed_samples: int,
        shard_paths: List[str],
    ) -> None:
        payload = {
            "dataset": self.config.CURRENT_NLP_DATASET,
            "algorithm": config_snapshot.get("algorithm"),
            "created_at": datetime.now().isoformat(),
            "total_samples": int(total_samples),
            "processed_samples": int(processed_samples),
            "shard_paths": shard_paths,
            "config": config_snapshot,
        }
        paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
        with paths["metadata"].open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        manifest_payload = {
            "shards": shard_paths,
            "updated_at": datetime.now().isoformat(),
        }
        with paths["manifest"].open("w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, ensure_ascii=False, indent=2)

    def _materialise_global_candidates(
        self,
        algorithm: str,
        candidate_bank: List[Dict[str, Any]],
        attention_analyzer: SelfAttentionAnalyzer,
        tokenizer,
    ) -> Tuple[List[str], List[int], List[str], List[str], List[Set[int]]]:
        algorithm_orientation = {
            'entropy_elbow_higher': True,
            'max_attention_elbow_lower': False,
            'removed_avg_elbow_higher': True,
            'top_k_avg_elbow_lower': False,
        }

        higher = algorithm_orientation.get(algorithm, True)
        score_tensors: List[torch.Tensor] = []
        for record in candidate_bank:
            scores = record.get('scores')
            if scores is None:
                continue
            if isinstance(scores, torch.Tensor):
                score_tensors.append(scores.to(torch.float32))
            else:
                score_tensors.append(torch.tensor(scores, dtype=torch.float32))

        if score_tensors:
            all_scores_tensor = torch.cat(score_tensors)
        else:
            all_scores_tensor = torch.tensor([], dtype=torch.float32)

        threshold_value, global_selected_count = attention_analyzer.compute_global_threshold(
            all_scores_tensor,
            higher=higher,
            context=algorithm,
        )

        total_score_count = all_scores_tensor.numel()
        if total_score_count > 0:
            threshold_display = (
                f"{threshold_value:.6f}" if threshold_value is not None else "None"
            )
            ratio_percent = (global_selected_count / total_score_count) * 100.0
            print(
                f"   🔎 Global selection summary: threshold={threshold_display}, "
                f"selected_words={global_selected_count}/{total_score_count} ({ratio_percent:.2f}%), "
                f"orientation={'higher' if higher else 'lower'}"
            )
        else:
            print("   🔎 Global selection summary: no valid word scores collected; using per-sample fallback")

        min_tokens = max(int(getattr(self.config, 'MIN_ATTENTION_TOP_TOKENS', 1)), 1)
        raw_top_k_limit = int(getattr(self.config, 'ATTENTION_TOP_K', 0))
        effective_top_k = max(raw_top_k_limit, min_tokens) if raw_top_k_limit > 0 else None

        # Masking is always enabled (Hendrycks requirement)
        mask_prob = float(getattr(self.config, 'MASKING_PROBABILITY', 0.3))

        rng = torch.Generator().manual_seed(getattr(self.config, 'OE_SAMPLING_SEED', getattr(self.config, 'RANDOM_STATE', 42)))

        oe_texts: List[str] = []
        oe_labels: List[int] = []
        original_texts: List[str] = []
        removed_words_list: List[str] = []
        masked_positions_per_sample: List[Set[int]] = []

        for sample_idx, record in enumerate(candidate_bank):
            scores_raw = record.get('scores')
            indices_raw = record.get('indices')
            spans_raw = record.get('spans', [])
            words_raw = record.get('words', [])

            if scores_raw is None or indices_raw is None:
                continue

            scores_tensor = scores_raw if isinstance(scores_raw, torch.Tensor) else torch.tensor(scores_raw, dtype=torch.float32)
            indices_tensor = indices_raw if isinstance(indices_raw, torch.Tensor) else torch.tensor(indices_raw, dtype=torch.long)

            if scores_tensor.numel() == 0 or indices_tensor.numel() == 0:
                continue

            if threshold_value is not None:
                if higher:
                    threshold_mask = scores_tensor >= threshold_value
                else:
                    threshold_mask = scores_tensor <= threshold_value
                selected_indices = indices_tensor[threshold_mask]
                selected_scores = scores_tensor[threshold_mask]
            else:
                selected_indices = indices_tensor.new_empty(0)
                selected_scores = scores_tensor.new_empty(0)

            if selected_indices.numel() == 0:
                fallback_count = min(min_tokens, indices_tensor.numel())
                if fallback_count == 0:
                    continue
                order = torch.argsort(scores_tensor, descending=higher)
                order = order[:fallback_count]
                selected_indices = indices_tensor[order]
                selected_scores = scores_tensor[order]

            if effective_top_k is not None and selected_indices.numel() > effective_top_k:
                order = torch.argsort(selected_scores, descending=higher)
                order = order[:effective_top_k]
                selected_indices = selected_indices[order]
                selected_scores = selected_scores[order]

            if selected_indices.numel() == 0:
                continue

            selected_positions = [int(idx.item()) for idx in selected_indices]
            spans = spans_raw
            words = words_raw

            input_ids_raw = record.get('input_ids')
            attention_mask_raw = record.get('attention_mask')
            if input_ids_raw is None or attention_mask_raw is None:
                continue

            input_ids_tensor = input_ids_raw if isinstance(input_ids_raw, torch.Tensor) else torch.tensor(input_ids_raw, dtype=torch.long)
            attention_mask_tensor = attention_mask_raw if isinstance(attention_mask_raw, torch.Tensor) else torch.tensor(attention_mask_raw, dtype=torch.long)

            input_ids_cpu = input_ids_tensor.clone()
            attention_mask_cpu = attention_mask_tensor

            masked_positions: Set[int] = set()
            removed_tokens: List[str] = []

            for pos_idx, global_index in enumerate(selected_positions):
                if global_index >= len(spans):
                    continue
                span = spans[global_index]
                word = words[global_index] if global_index < len(words) else ""

                if torch.rand(1, generator=rng).item() > mask_prob:
                    continue

                masked_any = False
                for token_pos in span:
                    if token_pos >= input_ids_cpu.size(0):
                        continue
                    if attention_mask_cpu[token_pos].item() == 0:
                        continue
                    input_ids_cpu[token_pos] = self.config.MASK_TOKEN_ID
                    masked_positions.add(int(token_pos))
                    masked_any = True
                if masked_any:
                    cleaned_word = word
                    if not cleaned_word:
                        cleaned_word = tokenizer.decode(
                            [int(input_ids_tensor[p].item()) for p in span if p < input_ids_tensor.size(0)],
                            skip_special_tokens=True
                        ).strip()
                    if cleaned_word:
                        removed_tokens.append(cleaned_word)

            if not masked_positions:
                continue

            try:
                oe_text = tokenizer.decode(input_ids_cpu, skip_special_tokens=True)
            except Exception:
                continue

            removed_words = ", ".join([w for w in removed_tokens if w]) if removed_tokens else ""

            oe_texts.append(oe_text)
            oe_labels.append(-1)
            original_texts.append(record.get('original_text', ''))
            removed_words_list.append(removed_words)
            masked_positions_per_sample.append(masked_positions)

            if attention_analyzer.debug_enabled:
                total_tokens = int(attention_mask_cpu.to(torch.int32).sum().item())
                attention_analyzer._maybe_log_filter_summary(
                    category=f"{algorithm}_global_summary",
                    batch_idx=sample_idx,
                    total_tokens=total_tokens,
                    selected_tokens=len(selected_positions),
                    extra={
                        'threshold': (
                            f"{threshold_value:.6f}" if threshold_value is not None else 'None'
                        ),
                        'top_k': effective_top_k if effective_top_k is not None else 'inf'
                    },
                )

        return oe_texts, oe_labels, original_texts, removed_words_list, masked_positions_per_sample

    # ------------------------------------------------------------------
    # Holdout helpers
    # ------------------------------------------------------------------

    def _filter_dataset_excluding(self, dataset, excluded_ids: set):
        if dataset is None:
            return
        keep_indices = [idx for idx, label in enumerate(dataset.labels) if label not in excluded_ids]
        dataset.texts = [dataset.texts[i] for i in keep_indices]
        dataset.labels = [dataset.labels[i] for i in keep_indices]

    def _apply_holdout_strategy(self, tokenizer):
        holdout_info = {}

        raw_data = NLPDatasetLoader.load_any_dataset(self.config.CURRENT_NLP_DATASET, split='train')
        if not raw_data or not raw_data.get('label'):
            raise ValueError("Holdout 모드를 사용하려면 라벨이 있는 데이터셋이 필요합니다.")

        label_map = self.data_module.label2id
        inverse_map = self.data_module.id2label
        class_ids = sorted({label_map[label] for label in raw_data['label']})

        holdout_count = max(
            self.config.OSR_HOLDOUT_MIN_CLASSES,
            int(len(class_ids) * self.config.OSR_HOLDOUT_RATIO)
        )
        holdout_count = min(holdout_count, len(class_ids) - 1) if len(class_ids) > 1 else 0
        if holdout_count <= 0:
            raise ValueError("Holdout 설정으로 인해 보류할 클래스가 없습니다. 비율 또는 최소 클래스를 조정하세요.")

        holdout_ids = set(class_ids[-holdout_count:])
        holdout_names = [inverse_map[cid] for cid in holdout_ids]

        holdout_texts = [
            text for text, label in zip(raw_data['text'], raw_data['label'])
            if label_map[label] in holdout_ids
        ]

        if not holdout_texts:
            raise ValueError("선택된 Holdout 클래스에 해당하는 샘플이 없습니다.")

        # Remove holdout classes from training/validation datasets
        self._filter_dataset_excluding(self.data_module.train_dataset, holdout_ids)
        self._filter_dataset_excluding(self.data_module.val_dataset, holdout_ids)
        self._filter_dataset_excluding(self.data_module.test_dataset, holdout_ids)

        # Remap labels to contiguous IDs
        remaining_ids = sorted({label for label in self.data_module.train_dataset.labels})
        remap = {old: idx for idx, old in enumerate(remaining_ids)}

        self.data_module.train_dataset.labels = [remap[label] for label in self.data_module.train_dataset.labels]
        self.data_module.val_dataset.labels = [remap[label] for label in self.data_module.val_dataset.labels]
        if self.data_module.test_dataset is not None:
            self.data_module.test_dataset.labels = [remap.get(label, label) for label in self.data_module.test_dataset.labels]

        self.data_module.num_classes = len(remaining_ids)
        self.data_module.label2id = {inverse_map[old]: new for old, new in remap.items()}
        self.data_module.id2label = {new: inverse_map[old] for old, new in remap.items()}

        holdout_dataset = OSRTextDataset(
            holdout_texts,
            [-1] * len(holdout_texts),
            tokenizer,
            self.config.OSR_MAX_LENGTH
        )

        holdout_info['dataset'] = holdout_dataset
        holdout_info['class_ids'] = list(holdout_ids)
        holdout_info['class_names'] = holdout_names

        self.logger.info(
            "Holdout mode enabled: excluded classes %s", ', '.join(holdout_names)
        )

        return holdout_info

    def _select_oe_texts(self, texts: List[str], limit: int, seed_key: str) -> List[str]:
        """Select a reproducible subset of texts up to the provided limit."""
        if not texts:
            return []

        if limit is None or limit <= 0 or len(texts) <= limit:
            return texts

        base_seed = getattr(self.config, 'OE_SAMPLING_SEED', 42)
        hashed = int(hashlib.md5(seed_key.encode('utf-8')).hexdigest(), 16) % 100000
        rng = random.Random(base_seed + hashed)
        indices = rng.sample(range(len(texts)), limit)
        indices.sort()
        return [texts[i] for i in indices]

    def run_baseline_training(self, dataset_name: str):
        """Train baseline model without OE"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"BASELINE TRAINING (No OE): {dataset_name}")
        self.logger.info(f"{'='*50}")
        
        self.config.setup_for_dataset(dataset_name)
        
        # Setup data
        self.logger.info("Setting up data module...")
        self.data_module = SimpleDataModule(self.config, dataset_name)
        self.data_module.setup()
        self.logger.info(f"Dataset setup complete - Train: {len(self.data_module.train_dataset)}, Val: {len(self.data_module.val_dataset)}, Classes: {self.data_module.num_classes}")
        
        # Calculate class weights if needed
        class_weights = None
        if self.config.USE_WEIGHTED_LOSS:
            train_labels = [self.data_module.train_dataset[i]['label'].item() 
                          for i in range(len(self.data_module.train_dataset))]
            class_counts = np.bincount(train_labels)
            class_weights = torch.FloatTensor(len(class_counts) / (len(class_counts) * class_counts))
            self.logger.info(f"Using class weights: {class_weights}")
        
        # Create model
        model = SimpleOEModel(
            config=self.config,
            num_labels=self.data_module.num_classes,
            label2id=self.data_module.label2id,
            id2label=self.data_module.id2label,
            class_weights=class_weights,
            tokenizer=self.data_module.tokenizer
        )
        
        # Setup trainer
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, mode='min')
        ]

        if not getattr(self.config, 'DISABLE_MODEL_CHECKPOINT', False):
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.config.MODEL_DIR,
                    filename=f'baseline_{dataset_name}',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                    save_last=True,
                    verbose=True
                )
            )

        if torch.cuda.is_available() and getattr(self.config, 'ENABLE_DEVICE_STATS_MONITOR', False):
            callbacks.append(DeviceStatsMonitor())
        
        # Create config dict for WandB
        baseline_config = {
            'dataset': dataset_name,
            'model_name': self.config.MODEL_NAME,
            'max_length': self.config.MAX_LENGTH,
            'batch_size': self.config.BATCH_SIZE,
            'learning_rate': self.config.LEARNING_RATE,
            'num_epochs': self.config.NUM_EPOCHS,
            'experiment_type': 'baseline',
            'use_amp': True,
            'use_sdpa': True
        }

        pytorch_logger = setup_pytorch_logger(
            log_dir=self.config.LOG_DIR,
            name=f"baseline_{dataset_name}",
            project="simplified-oe-baseline",
            config_dict=baseline_config,
            tags=['baseline', dataset_name, self.config.MODEL_NAME]
        )
        
        trainer = pl.Trainer(
            max_epochs=self.config.NUM_EPOCHS,
            callbacks=callbacks,
            logger=pytorch_logger,
            accelerator='auto',
            devices='auto',
            deterministic=True,
            precision='16-mixed',  # Enable AMP (Automatic Mixed Precision)
            enable_progress_bar=True,  # Disable progress bar for clean logs
            log_every_n_steps=50  # More frequent logging for better monitoring
        )
        
        # Train
        trainer.fit(model, self.data_module)
        
        # Test (only if test data is available)
        test_results = []
        if hasattr(self.data_module, 'test_dataloader') and self.data_module.test_dataloader() is not None:
            try:
                test_loader = self.data_module.test_dataloader()
                if test_loader and len(test_loader.dataset) > 0:
                    test_results = trainer.test(model, self.data_module)
                else:
                    print("No test data available, skipping test step")
            except Exception as e:
                print(f"Warning: Test step skipped due to error: {e}")
        else:
            print("No test dataloader available, skipping test step")
        print(f"Baseline test results: {test_results}")
        
        return model, trainer

    def _generate_attention_based_oe_data(self, id_train_samples, tokenizer, algorithm, model_state_dict=None, num_labels=None, sample_limit=None):
        """Generate attention-based OE data using specified algorithm"""

        dataset_name = self.config.CURRENT_NLP_DATASET
        total_input_samples = len(id_train_samples)
        max_samples_config = getattr(self.config, 'ATTENTION_GENERATION_MAX_SAMPLES', None)
        max_samples = max_samples_config if max_samples_config and max_samples_config > 0 else total_input_samples

        effective_limit = sample_limit if sample_limit and sample_limit > 0 else total_input_samples
        process_cap = min(max_samples, effective_limit)
        process_samples = max(1, min(total_input_samples, process_cap))

        random_seed = getattr(self.config, 'OE_SAMPLING_SEED', getattr(self.config, 'RANDOM_STATE', 42))
        rng = torch.Generator().manual_seed(random_seed)
        selected_indices = torch.randperm(total_input_samples, generator=rng)[:process_samples].tolist()
        sampled_train_samples = [id_train_samples[idx] for idx in selected_indices]

        generation_stats: Dict[str, Any] = {
            'algorithm': algorithm,
            'total_input_samples': total_input_samples,
            'process_cap': int(process_cap),
            'sample_limit': int(effective_limit),
            'processed_samples': 0,
            'generated_oe_samples': 0,
            'stage2_shards': 0,
        }

        print("\n📊 ===== OE 데이터셋 생성 시작 =====")
        print(f"   🎯 데이터셋: {dataset_name}")
        print(f"   🔧 알고리즘: {algorithm}")
        print(f"   📈 전체 ID 데이터셋 크기: {total_input_samples:,} samples")
        print(f"   🎲 처리할 샘플 수: {process_samples} samples (최대 {process_cap})")
        print("   ⚙️  설정:")
        print(f"      - ATTENTION_TOP_K: {self.config.ATTENTION_TOP_K}")
        print(f"      - ATTENTION_TOP_P: {self.config.ATTENTION_TOP_P}")
        print(f"      - MIN_ATTENTION_TOP_TOKENS: {self.config.MIN_ATTENTION_TOP_TOKENS}")
        print(f"      - USE_ATTENTION_ELBOW_REFINEMENT: {self.config.USE_ATTENTION_ELBOW_REFINEMENT}")
        print(f"      - MASKING_PROBABILITY: {self.config.MASKING_PROBABILITY}")
        print(f"      - ATTENTION_LAYER_INDEX: {self.config.ATTENTION_LAYER_INDEX}")
        print(f"      - ATTENTION_FILTERING_METHOD: {self.config.ATTENTION_FILTERING_METHOD}")
        print("📊 ==========================================\n")

        print(f"🧠 Generating attention-based OE data using {algorithm}...")

        generation_mode = getattr(self.config, 'ATTENTION_GENERATION_MODE', 'on_the_fly') or 'on_the_fly'
        stage_selection = getattr(self.config, 'ATTENTION_STAGE_TO_RUN', 'both') or 'both'
        staged_mode = generation_mode == 'staged'
        if stage_selection not in {'both', 'stage2', 'stage3'}:
            stage_selection = 'both'
        run_stage2 = (not staged_mode) or stage_selection in {'both', 'stage2'}
        run_stage3 = (not staged_mode) or stage_selection in {'both', 'stage3'}

        cache_paths = self._resolve_attention_cache_paths(algorithm)
        stage2_shard_size = int(getattr(self.config, 'ATTENTION_STAGE2_SHARD_SIZE', 512) or 512)
        stage2_shards: List[str] = []
        stage2_start_time: Optional[float] = None
        stage2_start_dt: Optional[datetime] = None
        if staged_mode and run_stage2:
            self._initialise_stage2_cache(cache_paths)

        try:
            from transformers import AutoModelForSequenceClassification
            temp_model = None
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if run_stage2:
                print(f"   🔄 Loading temp model: {self.config.MODEL_NAME}...")
                if num_labels is not None:
                    temp_model = AutoModelForSequenceClassification.from_pretrained(self.config.MODEL_NAME, num_labels=num_labels)
                else:
                    temp_model = AutoModelForSequenceClassification.from_pretrained(self.config.MODEL_NAME)
                if model_state_dict is not None:
                    missing, unexpected = temp_model.load_state_dict(model_state_dict, strict=False)
                    if missing:
                        print(f"   ⚠️ Missing keys while loading trained state: {missing[:5]}{'...' if len(missing) > 5 else ''}")
                    if unexpected:
                        print(f"   ⚠️ Unexpected keys while loading trained state: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
                temp_model.to(device)
                temp_model.eval()
                print("   ✅ Temp model loaded successfully")
            else:
                device = torch.device('cpu')

            print("   🔄 Initializing attention analyzer...")
            attention_analyzer = SelfAttentionAnalyzer(self.config, tokenizer)
            print("   ✅ Attention analyzer initialized")
        except Exception as exc:
            print(f"   ❌ Error in initialization: {exc}")
            return None, generation_stats

        global_selection_algorithms = {
            'entropy_elbow_higher',
            'max_attention_elbow_lower',
            'removed_avg_elbow_higher',
            'top_k_avg_elbow_lower',
        }
        use_global_selection = algorithm in global_selection_algorithms
        global_candidate_bank: List[Dict[str, Any]] = []

        oe_texts: List[str] = []
        oe_labels: List[int] = []
        original_texts: List[str] = []
        removed_words_list: List[str] = []
        masked_positions_per_sample: List[Set[int]] = []

        processed_samples = 0

        if run_stage2:
            batch_size = getattr(self.config, 'ATTENTION_GENERATION_BATCH_SIZE', 256)
            print(f"   🧮 Generation batch size: {batch_size}")
            total_samples = process_samples

            print(f"\n   🚀 Processing {total_samples} samples with batch_size={batch_size}...")

            stage2_start_time = time.time()
            stage2_start_dt = datetime.now()
            print(f"   ⏱️ Stage2 시작: {stage2_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

            num_batches = (total_samples + batch_size - 1) // batch_size
            current_shard_records: List[Dict[str, Any]] = []

            for batch_idx, offset in enumerate(range(0, total_samples, batch_size)):
                batch_samples = sampled_train_samples[offset:offset + batch_size]
                current_batch_size = len(batch_samples)

                progress = (batch_idx + 1) / num_batches * 100
                progress_bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
                print(
                    f"   ⏳ Batch {batch_idx + 1}/{num_batches} [{progress_bar}] {progress:.1f}% "
                    f"| Processing samples {processed_samples + 1}-{processed_samples + current_batch_size}"
                )

                input_ids = torch.stack([torch.tensor(s['input_ids']) for s in batch_samples]).to(device)
                attention_mask = torch.stack([torch.tensor(s['attention_mask']) for s in batch_samples]).to(device)

                attention_weights = attention_analyzer.extract_attention_weights(temp_model, input_ids, attention_mask)

                batch_oe_count = 0
                batch_candidate_count = 0

                if use_global_selection:
                    batch_scores_list, batch_indices_list, batch_spans_list, batch_words_list = attention_analyzer.compute_token_scores(
                        algorithm,
                        attention_weights,
                        attention_mask,
                        input_ids,
                    )
                else:
                    batch_scores_list = batch_indices_list = batch_spans_list = batch_words_list = None

                for j, sample in enumerate(batch_samples):
                    sample_attention = attention_weights[j]
                    sample_input_ids = input_ids[j]
                    sample_attention_mask = attention_mask[j]
                    original_text = tokenizer.decode(sample_input_ids.cpu(), skip_special_tokens=True)

                    if use_global_selection:
                        assert batch_scores_list is not None and batch_indices_list is not None
                        scores_tensor = batch_scores_list[j]
                        indices_tensor = batch_indices_list[j]
                        spans_list = batch_spans_list[j]
                        words_list = batch_words_list[j]

                        if scores_tensor.numel() == 0 or indices_tensor.numel() == 0:
                            continue

                        record = {
                            'input_ids': sample_input_ids.detach().cpu().tolist(),
                            'attention_mask': sample_attention_mask.detach().cpu().tolist(),
                            'scores': scores_tensor.detach().cpu().to(torch.float32).tolist(),
                            'indices': indices_tensor.detach().cpu().to(torch.long).tolist(),
                            'spans': [[int(idx) for idx in span] for span in spans_list],
                            'words': [word for word in words_list],
                            'original_text': original_text,
                        }

                        if staged_mode:
                            current_shard_records.append(record)
                            if len(current_shard_records) >= stage2_shard_size:
                                shard_path = self._write_stage2_shard(current_shard_records, len(stage2_shards), cache_paths)
                                stage2_shards.append(str(shard_path))
                                current_shard_records = []
                            if stage_selection == 'both':
                                global_candidate_bank.append(record)
                        else:
                            global_candidate_bank.append(record)

                        batch_candidate_count += 1
                        continue

                    modified_input_ids, removed_tokens, masked_positions = self._apply_attention_algorithm(
                        algorithm,
                        sample_input_ids,
                        sample_attention_mask,
                        sample_attention,
                        attention_analyzer,
                        tokenizer,
                    )

                    try:
                        oe_text = tokenizer.decode(modified_input_ids.cpu(), skip_special_tokens=True)
                    except Exception:
                        continue

                    removed_words = ", ".join(removed_tokens) if removed_tokens else ""
                    oe_texts.append(oe_text)
                    oe_labels.append(-1)
                    original_texts.append(original_text)
                    removed_words_list.append(removed_words)
                    masked_positions_per_sample.append(masked_positions)
                    batch_oe_count += 1

                processed_samples += current_batch_size
                if use_global_selection:
                    print(f"      ✅ Batch {batch_idx + 1} collected {batch_candidate_count} candidate score sets")
                else:
                    print(f"      ✅ Batch {batch_idx + 1} completed: {batch_oe_count} OE samples generated")

            if staged_mode and current_shard_records:
                shard_path = self._write_stage2_shard(current_shard_records, len(stage2_shards), cache_paths)
                stage2_shards.append(str(shard_path))
            generation_stats['stage2_shards'] = len(stage2_shards)
        else:
            print("   ⏭️  Stage2 (attention extraction) skipped per configuration.")

        if staged_mode and run_stage2:
            config_snapshot = {
                'algorithm': algorithm,
                'attention_top_p': getattr(self.config, 'ATTENTION_TOP_P', None),
                'attention_top_k': getattr(self.config, 'ATTENTION_TOP_K', None),
                'min_attention_top_tokens': getattr(self.config, 'MIN_ATTENTION_TOP_TOKENS', None),
                'use_elbow': getattr(self.config, 'USE_ATTENTION_ELBOW_REFINEMENT', True),
                'masking_probability': getattr(self.config, 'MASKING_PROBABILITY', None),
                'generation_mode': generation_mode,
                'stage_selection': stage_selection,
            }
            self._write_stage2_metadata(
                cache_paths,
                config_snapshot=config_snapshot,
                total_samples=total_input_samples,
                processed_samples=processed_samples,
                shard_paths=stage2_shards,
            )
            print(f"   💾 Stage2 cache stored under {cache_paths['stage2_dir']}")

            if stage2_start_time is not None:
                stage2_end_dt = datetime.now()
                stage2_duration = time.time() - stage2_start_time
                print(
                    f"   ⏱️ Stage2 종료: {stage2_end_dt.strftime('%Y-%m-%d %H:%M:%S')}"
                    f" (총 {stage2_duration:.2f}초)"
                )
                generation_stats['stage2_start'] = stage2_start_dt.isoformat() if stage2_start_dt else None
                generation_stats['stage2_end'] = stage2_end_dt.isoformat()
                generation_stats['stage2_duration_seconds'] = stage2_duration

        if staged_mode and not run_stage3:
            print("   ✅ Stage2 completed. Stage3 generation skipped as requested.")
            generation_stats['processed_samples'] = int(processed_samples)
            generation_stats['generated_oe_samples'] = len(oe_texts)
            if 'temp_model' in locals() and temp_model is not None:
                del temp_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return None, generation_stats

        if staged_mode and use_global_selection and stage_selection == 'stage3':
            try:
                global_candidate_bank = self._load_stage2_candidates(cache_paths)
            except FileNotFoundError as exc:
                print(f"   ❌ Stage3 aborted: {exc}")
                return None
        elif staged_mode and stage_selection == 'stage3' and not use_global_selection:
            print("   ⚠️ Stage3 requested but this algorithm does not support global selection; no new samples will be produced.")

        if processed_samples == 0 and use_global_selection and global_candidate_bank:
            processed_samples = len(global_candidate_bank)
        generation_stats['processed_samples'] = int(processed_samples)

        if use_global_selection:
            print("\n   ⚙️  Global selection pending - processing collected scores...")
            oe_texts_global, oe_labels_global, original_texts_global, removed_words_global, masked_positions_global = self._materialise_global_candidates(
                algorithm,
                global_candidate_bank,
                attention_analyzer,
                tokenizer,
            )
            oe_texts.extend(oe_texts_global)
            oe_labels.extend(oe_labels_global)
            original_texts.extend(original_texts_global)
            removed_words_list.extend(removed_words_global)
            masked_positions_per_sample.extend(masked_positions_global)
            print(f"   ✅ Global selection applied: {len(oe_texts_global)} OE samples generated")

        print(f"\n   🎯 총 처리 완료: {processed_samples} samples → {len(oe_texts)} OE samples 생성")

        print("\n📊 ===== OE 데이터셋 생성 완료 =====")
        print(f"   🎯 데이터셋: {dataset_name}")
        print(f"   🔧 알고리즘: {algorithm}")
        print("   📊 결과:")
        print(f"      - 전체 ID 데이터셋: {total_input_samples:,} samples")
        print(f"      - 처리한 샘플: {processed_samples} samples")
        print(f"      - 생성된 OE 샘플: {len(oe_texts)} samples")
        ratio = len(oe_texts) / processed_samples if processed_samples else 0.0
        print(f"      - 생성 비율: {ratio:.2f} OE/input")

        csv_path, pt_path = self._save_extracted_oe_dataset(
            oe_texts,
            oe_labels,
            algorithm,
            original_texts,
            removed_words_list,
            masked_positions_per_sample,
        )

        print("   💾 저장 위치:")
        print(f"      - CSV: {csv_path}")
        print(f"      - PyTorch: {pt_path}")
        print("📊 ========================================\n")

        if 'temp_model' in locals() and temp_model is not None:
            del temp_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        oe_dataset = OSRTextDataset(
            oe_texts,
            oe_labels,
            tokenizer,
            self.config.OSR_MAX_LENGTH,
        )

        generation_stats['generated_oe_samples'] = len(oe_dataset)
        return oe_dataset, generation_stats

    def _save_extracted_oe_dataset(self, oe_texts, oe_labels, algorithm, original_texts=None, removed_words_list=None, masked_positions_per_sample=None):
        """Save extracted OE dataset to files"""
        import pandas as pd
        import os

        # Create extracted_oe_datasets directory
        oe_dir = os.path.join(self.config.OUTPUT_DIR, 'extracted_oe_datasets')
        os.makedirs(oe_dir, exist_ok=True)

        # Current dataset name
        dataset_name = self.config.CURRENT_NLP_DATASET
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save as CSV with new structure
        df_data = {
            'text': oe_texts,
            'label': oe_labels,
            'algorithm': [algorithm] * len(oe_texts)
        }

        # Add original text and removed words if available
        if original_texts is not None:
            df_data['original_text'] = original_texts
        if removed_words_list is not None:
            df_data['removed_words'] = removed_words_list

        df = pd.DataFrame(df_data)

        if masked_positions_per_sample is not None:
            has_mask = [len(masked_positions_per_sample[idx]) > 0 for idx in range(len(masked_positions_per_sample))]
            df['has_mask'] = has_mask
        else:
            df['has_mask'] = True

        # Filter out rows where masked text equals original text
        if original_texts is not None and 'original_text' in df.columns:
            same_as_original = df['text'] == df['original_text']
            df = df.loc[~same_as_original]

        # Filter out rows where no masked positions were recorded
        if 'has_mask' in df.columns:
            mask = df['has_mask'].astype(bool)
            df = df.loc[mask.values]

        df_to_save = df.drop(columns=['has_mask'], errors='ignore')
        csv_path = os.path.join(oe_dir, f'oe_{dataset_name}_{algorithm}_{timestamp}.csv')
        df_to_save.to_csv(csv_path, index=False)

        # Save as PyTorch tensor with enhanced data
        filtered_texts = df_to_save['text'].tolist()
        filtered_labels = df_to_save['label'].tolist()

        pt_data = {
            'texts': filtered_texts,
            'labels': torch.tensor(filtered_labels),
            'algorithm': algorithm,
            'dataset': dataset_name,
            'timestamp': timestamp,
            'config': {
                'attention_top_k': self.config.ATTENTION_TOP_K,
                'attention_top_p': self.config.ATTENTION_TOP_P,
                'min_attention_top_tokens': self.config.MIN_ATTENTION_TOP_TOKENS,
                'use_attention_elbow_refinement': self.config.USE_ATTENTION_ELBOW_REFINEMENT,
                'masking_probability': self.config.MASKING_PROBABILITY,
                'attention_layer_index': self.config.ATTENTION_LAYER_INDEX,
                'attention_filtering_method': self.config.ATTENTION_FILTERING_METHOD
            }
        }

        # Add original data if available
        if original_texts is not None and 'original_text' in df_to_save.columns:
            pt_data['original_texts'] = df_to_save['original_text'].tolist()
        if removed_words_list is not None and 'removed_words' in df_to_save.columns:
            pt_data['removed_words'] = df_to_save['removed_words'].tolist()
        if masked_positions_per_sample is not None:
            pt_data['masked_positions'] = [masked_positions_per_sample[idx] for idx in df_to_save.index]

        pt_path = os.path.join(oe_dir, f'oe_{dataset_name}_{algorithm}_{timestamp}.pt')
        torch.save(pt_data, pt_path)

        # Also save original dataset separately
        original_csv_path = self._save_original_dataset(dataset_name, original_texts, oe_labels)

        return csv_path, pt_path

    def _save_original_dataset(self, dataset_name, original_texts, labels):
        """Save original dataset separately for reference"""
        import pandas as pd
        import os

        # Create extracted_oe_datasets directory
        oe_dir = os.path.join(self.config.OUTPUT_DIR, 'extracted_oe_datasets')
        os.makedirs(oe_dir, exist_ok=True)

        # Save original dataset
        if original_texts is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_df = pd.DataFrame({
                'text': original_texts,
                'label': labels
            })

            original_csv_path = os.path.join(oe_dir, f'original_{dataset_name}_{timestamp}.csv')
            original_df.to_csv(original_csv_path, index=False)

            print(f"      - Original dataset: {original_csv_path}")
            return original_csv_path

        return None

    def _apply_attention_algorithm(self, algorithm, input_ids, attention_mask,
                                 attention_weights, attention_analyzer, tokenizer):
        """Apply specific attention algorithm to generate OE data"""

        # Get valid token positions (excluding padding)
        valid_positions = attention_mask.bool()
        seq_len = valid_positions.sum().item()
        removed_tokens = []
        masked_positions: Set[int] = set()

        top_p_config = float(getattr(self.config, 'ATTENTION_TOP_P', 0.0) or 0.0)
        min_tokens_config = int(getattr(self.config, 'MIN_ATTENTION_TOP_TOKENS', 0) or 0)
        use_elbow = bool(getattr(self.config, 'USE_ATTENTION_ELBOW_REFINEMENT', True))

        threshold_value: Optional[float] = None
        elbow_idx_value: Optional[int] = None
        stat_payload: Dict[str, float] = {}

        if algorithm == 'entropy_elbow_higher':
            # Calculate attention entropy and find elbow point
            entropy = attention_analyzer.calculate_attention_entropy(attention_weights[:seq_len, :seq_len])
            elbow_idx = attention_analyzer.find_elbow_point(entropy, higher=True)

            # Get threshold value for filtering
            sorted_entropy, _ = torch.sort(entropy, descending=True)
            threshold = sorted_entropy[elbow_idx] if elbow_idx < len(sorted_entropy) else sorted_entropy[-1]
            mask_positions = torch.nonzero(entropy >= threshold, as_tuple=False).flatten()

            # Log entropy filtering criteria
            entropy_mean = entropy.mean().item()
            entropy_std = entropy.std().item()
            entropy_min = entropy.min().item()
            entropy_max = entropy.max().item()

            selected_count_preview = int(mask_positions.numel())
            selected_ratio_preview = selected_count_preview / seq_len if seq_len > 0 else 0.0

            print(f"   🔍 Entropy Filtering Criteria:")
            print(f"      - Algorithm: {algorithm}")
            print(f"      - Sequence length: {seq_len}")
            print(f"      - Entropy statistics: mean={entropy_mean:.4f}, std={entropy_std:.4f}")
            print(f"      - Entropy range: [{entropy_min:.4f}, {entropy_max:.4f}]")
            print(f"      - Elbow point index: {elbow_idx}/{len(entropy)}")
            print(f"      - Threshold value: {threshold:.4f}")
            print(
                f"      - Tokens above threshold: {selected_count_preview}/{seq_len} "
                f"({selected_ratio_preview * 100:.2f}%)"
            )
            print(
                f"      - top_p_config={top_p_config:.4f}, min_tokens_config={min_tokens_config}, "
                f"use_elbow={use_elbow}"
            )

            threshold_value = float(threshold.detach().cpu().item())
            elbow_idx_value = int(elbow_idx)
            stat_payload = {
                'mean': entropy_mean,
                'std': entropy_std,
                'min': entropy_min,
                'max': entropy_max,
            }

        elif algorithm == 'max_attention_elbow_lower':
            # Find maximum attention values and elbow point
            max_attention = attention_weights[:seq_len, :seq_len].max(dim=-1)[0]
            elbow_idx = attention_analyzer.find_elbow_point(max_attention, higher=False)

            # Get threshold value for filtering (lower values)
            sorted_max_attention, _ = torch.sort(max_attention, descending=False)
            threshold = sorted_max_attention[elbow_idx] if elbow_idx < len(sorted_max_attention) else sorted_max_attention[-1]
            mask_positions = torch.nonzero(max_attention <= threshold, as_tuple=False).flatten()

            threshold_value = float(threshold.detach().cpu().item())
            elbow_idx_value = int(elbow_idx)
            stat_payload = {
                'mean': max_attention.mean().item(),
                'std': max_attention.std().item(),
                'min': max_attention.min().item(),
                'max': max_attention.max().item(),
            }

        elif algorithm == 'removed_avg_elbow_higher':
            # Calculate average attention (excluding self-attention)
            attention_no_self = attention_weights[:seq_len, :seq_len].clone()
            attention_no_self.fill_diagonal_(0)
            avg_attention = attention_no_self.mean(dim=-1)
            elbow_idx = attention_analyzer.find_elbow_point(avg_attention, higher=True)

            # Get threshold value for filtering (higher values)
            sorted_avg_attention, _ = torch.sort(avg_attention, descending=True)
            threshold = sorted_avg_attention[elbow_idx] if elbow_idx < len(sorted_avg_attention) else sorted_avg_attention[-1]
            mask_positions = torch.nonzero(avg_attention >= threshold, as_tuple=False).flatten()

            threshold_value = float(threshold.detach().cpu().item())
            elbow_idx_value = int(elbow_idx)
            stat_payload = {
                'mean': avg_attention.mean().item(),
                'std': avg_attention.std().item(),
                'min': avg_attention.min().item(),
                'max': avg_attention.max().item(),
            }

        elif algorithm == 'top_k_avg_elbow_lower':
            # Calculate top-k average attention
            top_k = min(self.config.ATTENTION_TOP_K, seq_len)
            top_k_attention = attention_weights[:seq_len, :seq_len].topk(top_k, dim=-1)[0].mean(dim=-1)
            elbow_idx = attention_analyzer.find_elbow_point(top_k_attention, higher=False)

            # Get threshold value for filtering (lower values)
            sorted_top_k_attention, _ = torch.sort(top_k_attention, descending=False)
            threshold = sorted_top_k_attention[elbow_idx] if elbow_idx < len(sorted_top_k_attention) else sorted_top_k_attention[-1]

            # Only select tokens below threshold (lower top-k average attention)
            mask_positions = torch.nonzero(top_k_attention <= threshold, as_tuple=False).flatten()

        elif algorithm == 'sequential':
            # Sequential masking based on position and attention
            attention_scores = attention_weights[:seq_len, :seq_len].mean(dim=-1)
            # Mask every other high-attention token
            sorted_indices = torch.argsort(attention_scores, descending=True)
            mask_positions = sorted_indices[::2][:seq_len//4]  # Mask 25% of tokens

        else:
            # Default: random masking
            mask_positions = torch.randperm(seq_len)[:seq_len//4]

        selected_count = int(mask_positions.numel()) if torch.is_tensor(mask_positions) else len(mask_positions)
        selected_ratio = selected_count / seq_len if seq_len > 0 else 0.0

        if getattr(attention_analyzer, 'debug_enabled', False):
            if attention_analyzer._should_log_debug(f"algorithm_summary_{algorithm}"):
                ratio_percent = selected_ratio * 100.0
                threshold_display = f"{threshold_value:.6f}" if threshold_value is not None else "n/a"
                print(
                    "   🔎 Attention Debug Summary: "
                    f"algorithm={algorithm}, seq_len={seq_len}, selected={selected_count} ({ratio_percent:.2f}%), "
                    f"top_p={top_p_config:.4f}, min_tokens={min_tokens_config}, "
                    f"elbow_idx={elbow_idx_value}, threshold={threshold_display}, use_elbow={use_elbow}"
                )
            summary_stats = ", ".join(
                f"{key}={value:.6f}" for key, value in stat_payload.items()
            ) if stat_payload else ""
            attention_analyzer._log_debug(
                f"[algorithm_summary][{algorithm}] seq_len={seq_len}, selected={selected_count}, "
                f"ratio={selected_ratio:.6f}, top_p={top_p_config:.4f}, min_tokens={min_tokens_config}, "
                f"elbow_idx={elbow_idx_value}, threshold={threshold_value}, use_elbow={use_elbow}, stats="
                f"[{summary_stats}]"
            )

        # Apply masking and track removed tokens
        modified_input_ids = input_ids.clone()
        for pos in mask_positions:
            if pos < len(modified_input_ids) and attention_mask[pos]:
                # Apply masking with probability
                if torch.rand(1).item() < self.config.MASKING_PROBABILITY:
                    # Get the original token before masking
                    original_token_id = input_ids[pos].item()
                    original_token = tokenizer.decode([original_token_id], skip_special_tokens=True).strip()
                    if original_token:  # Only add non-empty tokens
                        removed_tokens.append(original_token)

                    modified_input_ids[pos] = self.config.MASK_TOKEN_ID
                    masked_positions.add(int(pos))

        return modified_input_ids, removed_tokens, masked_positions

    def run_osr_experiments(self, dataset_name: str):
        """Run OSR experiments with different OE sources"""
        print(f"\n{'='*50}")
        print(f"OSR EXPERIMENTS: {dataset_name}")
        print(f"{'='*50}")
        self.logger.info(f"== OSR Experiments start :: dataset={dataset_name}, mode={self.config.OE_METHOD}, holdout={self.config.OSR_EXPERIMENT_METHOD}")

        # Reset seeds so repeated invocations (e.g., traditional → self-attention) start from identical RNG state
        base_seed = getattr(self.config, 'RANDOM_STATE', 42)
        set_seed(int(base_seed))

        self.config.setup_for_dataset(dataset_name)
        
        # Setup data
        self.data_module = SimpleDataModule(self.config, dataset_name)
        self.data_module.setup()

        stage_selection = getattr(self.config, 'ATTENTION_STAGE_TO_RUN', 'both') or 'both'
        stage3_only_request = (
            self.config.ATTENTION_GENERATION_MODE == 'staged'
            and stage_selection == 'stage3'
        )
        reuse_standard_model = bool(getattr(self.config, 'STAGE3_USE_CACHED_STANDARD_MODEL', False))
        dataset_cache_dir = Path(self.config.ATTENTION_CACHE_DIR) / dataset_name
        default_checkpoint_path = dataset_cache_dir / 'standard_model.pt'
        configured_checkpoint = getattr(self.config, 'STAGE3_STANDARD_CHECKPOINT', None)
        checkpoint_path = Path(configured_checkpoint) if configured_checkpoint else default_checkpoint_path
        checkpoint_exists = checkpoint_path.exists()
        if reuse_standard_model:
            self.config.STAGE3_STANDARD_CHECKPOINT = str(checkpoint_path)
        skip_standard_training = stage3_only_request and reuse_standard_model and checkpoint_exists
        if stage3_only_request and reuse_standard_model and not checkpoint_exists:
            print(f"⚠️  Requested reuse of cached Standard model, but checkpoint not found at {checkpoint_path}. Retraining Standard scenario.")
            skip_standard_training = False
        self._stage3_checkpoint_path = checkpoint_path
        self.standard_model_state = None
        if skip_standard_training and checkpoint_exists:
            try:
                self.standard_model_state = torch.load(str(checkpoint_path), map_location='cpu')
                print(f"✅ Loaded cached Standard checkpoint from {checkpoint_path}")
            except Exception as exc:
                print(f"⚠️  Failed to load cached Standard checkpoint: {exc}")
        
        tokenizer = self.data_module.tokenizer
        num_classes = self.data_module.num_classes

        holdout_info = None
        if self.config.OSR_EXPERIMENT_METHOD == 'holdout':
            holdout_info = self._apply_holdout_strategy(tokenizer)
            num_classes = self.data_module.num_classes

        # Create label mappings for the model
        label2id = {str(i): i for i in range(num_classes)}
        id2label = {i: str(i) for i in range(num_classes)}
        
        # Prepare ID data loaders (after potential holdout filtering)
        id_train_loader = self.data_module.train_dataloader()
        id_test_loader = self.data_module.val_dataloader()  # Use validation data since test is empty

        # Prepare OE scenarios and OOD datasets based on OE_METHOD
        oe_scenarios = []
        ood_datasets = {}

        print(f"🔧 OE Method: {self.config.OE_METHOD}")
        self.logger.info(f"OE method: {self.config.OE_METHOD}")

        stage2_only_request = (
            self.config.ATTENTION_GENERATION_MODE == 'staged'
            and self.config.ATTENTION_STAGE_TO_RUN == 'stage2'
            and self.config.USE_SELF_ATTENTION_OE
        )

        # Get full training data for ID/OE scenario generation
        id_train_samples = list(self.data_module.train_dataset)
        id_train_sample_count = len(id_train_samples)

        configured_limit = getattr(self.config, 'OE_MAX_SAMPLES', None)
        if configured_limit and configured_limit > 0:
            oe_sample_limit = min(configured_limit, id_train_sample_count)
        else:
            oe_sample_limit = id_train_sample_count
        oe_sample_limit = max(1, oe_sample_limit)

        self.current_oe_sample_limit = oe_sample_limit
        self.id_sample_count = id_train_sample_count

        print(f"🎯 OE sample limit per scenario: {oe_sample_limit} samples")
        self.logger.info(f"OE sample limit per scenario: {oe_sample_limit}")

        # 1. Standard (no OE) - include unless we are reusing a cached checkpoint for staged stage3-only runs
        if skip_standard_training:
            print("⚡️  Skipping Standard retraining: using cached Standard checkpoint for stage3 sweep.")
        else:
            oe_scenarios.append({
                'tag': 'Standard',
                'dataset': None,
                'algorithm': None,
                'needs_generation': False,
                'oe_sample_count': 0,
                'sample_limit': oe_sample_limit
            })

        # 2. Self-Attention OE scenarios (pre-generated datasets)
        if self.config.OE_METHOD in ['self_attention', 'both']:
            attention_algorithms = [
                'entropy_elbow_higher',
                'max_attention_elbow_lower',
                'removed_avg_elbow_higher',
                'top_k_avg_elbow_lower',
                'sequential'
            ]


            print(f"🧠 Self-Attention OE: generating datasets for {len(attention_algorithms)} algorithms...")
            for algorithm in attention_algorithms:
                oe_scenarios.append({
                    'tag': f"Self_Attention_OE_{algorithm}",
                    'dataset': None,
                    'algorithm': algorithm,
                    'needs_generation': True,
                    'oe_sample_count': 0,
                    'sample_limit': oe_sample_limit
                })

        # 3. Traditional OE scenarios
        if self.config.OE_METHOD in ['traditional', 'both'] and self.config.OSR_EXPERIMENT_METHOD != 'holdout':
            print(f"📚 Traditional OE: Using {len(self.config.OSR_EXTERNAL_OE_DATASETS)} external datasets...")
            for oe_key in self.config.OSR_EXTERNAL_OE_DATASETS:
                if oe_key == dataset_name:  # Skip self-reference
                    continue

                oe_data = NLPDatasetLoader.load_any_dataset(oe_key, split='train')
                if oe_data and oe_data['text']:
                    sampled_texts = self._select_oe_texts(
                        oe_data['text'],
                        oe_sample_limit,
                        f"traditional_{dataset_name}_{oe_key}"
                    )

                    oe_ds = OSRTextDataset(
                        sampled_texts,
                        [-1] * len(sampled_texts),  # -1 for OE samples
                        tokenizer,
                        self.config.OSR_MAX_LENGTH
                    )
                    oe_scenarios.append({
                        'tag': f"OE_{oe_key}",
                        'dataset': oe_ds,
                        'algorithm': None,
                        'needs_generation': False,
                        'oe_sample_count': len(sampled_texts),
                        'sample_limit': oe_sample_limit
                    })

        # Prepare external OOD test datasets for evaluation
        print(f"📊 Setting up OOD evaluation datasets...")
        if self.config.OSR_EXPERIMENT_METHOD == 'holdout' and holdout_info:
            ood_datasets['holdout'] = holdout_info['dataset']
        else:
            for ood_key in self.config.OSR_EXTERNAL_OE_DATASETS:
                if ood_key == dataset_name:  # Skip self-reference
                    continue

                ood_data = NLPDatasetLoader.load_any_dataset(ood_key, split='test')
                if not ood_data:
                    ood_data = NLPDatasetLoader.load_any_dataset(ood_key, split='train')

                if ood_data and ood_data['text']:
                    # Sample subset for evaluation
                    eval_limit = min(
                        getattr(self.config, 'OOD_EVAL_MAX_SAMPLES', 1000),
                        self.current_oe_sample_limit
                    )

                    sampled_ood_texts = self._select_oe_texts(
                        ood_data['text'],
                        eval_limit,
                        f"ood_eval_{dataset_name}_{ood_key}"
                    )

                    ood_ds = OSRTextDataset(
                        sampled_ood_texts,
                        [-1] * len(sampled_ood_texts),
                        tokenizer,
                        self.config.OSR_MAX_LENGTH
                    )
                    ood_datasets[ood_key] = ood_ds
        
        # Train and evaluate each OE scenario
        results = []
        generated_self_attention_datasets: Dict[str, OSRTextDataset] = {}
        generated_self_attention_stats: Dict[str, Dict[str, Any]] = {}

        print(f"\n{'='*60}")
        print(f"🚀 SEQUENTIAL OSR EXPERIMENT PIPELINE")
        print(f"📊 Total models to train: {len(oe_scenarios)}")
        print(f"🎯 Dataset: {dataset_name}")
        print(f"🔧 Method: {self.config.OE_METHOD}")
        print(f"{'='*60}")
        self.logger.info(f"Total OSR scenarios: {len(oe_scenarios)}")

        for scenario_idx, scenario in enumerate(oe_scenarios, 1):
            # Ensure each scenario starts from a deterministic seed (Standard gets identical seed across modes)
            scenario_seed = int(base_seed) + scenario_idx - 1
            set_seed(scenario_seed)

            oe_tag = scenario['tag']
            algorithm = scenario.get('algorithm')
            oe_dataset = scenario.get('dataset')
            if scenario.get('needs_generation') and oe_dataset is None and algorithm in generated_self_attention_datasets:
                oe_dataset = generated_self_attention_datasets[algorithm]
                scenario_stats_cached = generated_self_attention_stats.get(algorithm, {})
                scenario['dataset'] = oe_dataset
                scenario['oe_sample_count'] = len(oe_dataset) if oe_dataset is not None else 0
                scenario['oe_processed_count'] = scenario_stats_cached.get('processed_samples', scenario['oe_sample_count'])
                scenario['oe_total_input'] = scenario_stats_cached.get('total_input_samples', id_train_sample_count)
                scenario['oe_generation_stats'] = scenario_stats_cached
            elif scenario.get('needs_generation') and oe_dataset is None and algorithm:
                # NEW: For Stage3 with staged mode, use CachedAttentionOEDataset for on-the-fly generation
                if stage3_only_request:
                    print(f"   ⏳ Creating CachedAttentionOEDataset for algorithm {algorithm} ...")
                    try:
                        # Resolve cache directory using centralized function
                        cache_dir = resolve_attention_cache_path(
                            base_dir=Path(self.config.ATTENTION_CACHE_DIR),
                            dataset=dataset_name,
                            algorithm=algorithm,
                            top_p=getattr(self.config, 'ATTENTION_TOP_P', None)
                        )

                        # Check if Stage2 cache exists
                        if not (cache_dir / "metadata.json").exists():
                            raise FileNotFoundError(
                                f"Stage2 cache not found at {cache_dir}. "
                                f"Run Stage2 first with --attention_stage stage2"
                            )

                        # Create CachedAttentionOEDataset
                        generated_ds = CachedAttentionOEDataset(
                            cache_dir=cache_dir,
                            tokenizer=tokenizer,
                            config=self.config,
                            max_length=self.config.OSR_MAX_LENGTH
                        )

                        # Load metadata for statistics
                        with open(cache_dir / "metadata.json", 'r') as f:
                            metadata = json.load(f)

                        generation_stats = {
                            'algorithm': algorithm,
                            'total_input_samples': metadata.get('total_samples', len(generated_ds)),
                            'processed_samples': metadata.get('processed_samples', len(generated_ds)),
                            'generated_oe_samples': len(generated_ds),
                            'stage2_shards': len(list((cache_dir / "stage2").glob("shard_*.pt"))),
                            'cache_mode': 'on_the_fly'
                        }

                        generated_self_attention_datasets[algorithm] = generated_ds
                        generated_self_attention_stats[algorithm] = generation_stats
                        scenario['dataset'] = generated_ds
                        scenario['oe_sample_count'] = len(generated_ds)
                        scenario['oe_processed_count'] = generation_stats.get('processed_samples', len(generated_ds))
                        scenario['oe_total_input'] = generation_stats.get('total_input_samples', id_train_sample_count)
                        scenario['oe_generation_stats'] = generation_stats
                        oe_dataset = generated_ds
                        print(f"   ✅ CachedAttentionOEDataset created: {len(generated_ds)} samples available")
                    except FileNotFoundError as exc:
                        print(f"   ❌ {exc}")
                        scenario['dataset'] = None
                        oe_dataset = None
                    except Exception as exc:
                        print(f"   ⚠️ Failed to create CachedAttentionOEDataset: {exc}")
                        import traceback
                        traceback.print_exc()
                        scenario['dataset'] = None
                        oe_dataset = None
                else:
                    # OLD: For non-staged or stage2+stage3 together, materialize OE dataset
                    print(f"   ⏳ Loading cached Stage2 data for algorithm {algorithm} ...")
                    generated_ds, generation_stats = self._generate_attention_based_oe_data(
                        id_train_samples,
                        tokenizer,
                        algorithm,
                        model_state_dict=None,
                        num_labels=self.data_module.num_classes,
                        sample_limit=getattr(self, 'current_oe_sample_limit', None)
                    )
                    scenario['oe_generation_stats'] = generation_stats
                    if generated_ds and len(generated_ds) > 0:
                        generated_self_attention_datasets[algorithm] = generated_ds
                        generated_self_attention_stats[algorithm] = generation_stats
                        scenario['dataset'] = generated_ds
                        scenario['oe_sample_count'] = len(generated_ds)
                        scenario['oe_processed_count'] = generation_stats.get('processed_samples', len(generated_ds))
                        scenario['oe_total_input'] = generation_stats.get('total_input_samples', id_train_sample_count)
                        oe_dataset = generated_ds
                        print(f"   ✅ Loaded Stage3 OE dataset for {algorithm} from cache")
                    else:
                        print(f"   ⚠️ Failed to materialise OE dataset for {algorithm} using cached Stage2 data")

            scenario_algorithm = scenario.get('algorithm')
            previous_filter_method = self.config.ATTENTION_FILTERING_METHOD
            if scenario_algorithm:
                self.config.ATTENTION_FILTERING_METHOD = scenario_algorithm

            # Determine if Self-Attention should be enabled for this scenario
            # Enable Self-Attention for attention-based OE models
            enable_self_attention = oe_tag.startswith('Self_Attention_OE_')
            scenario_type = (
                'SelfAttention' if enable_self_attention else
                ('Traditional' if oe_dataset is not None else 'Standard')
            )
            scenario_sample_count = scenario.get('oe_sample_count', len(oe_dataset) if oe_dataset else 0)
            scenario_processed_count = scenario.get('oe_processed_count', scenario_sample_count)
            scenario_total_input = scenario.get('oe_total_input', id_train_sample_count)

            # Progress logging
            current_progress = f"({scenario_idx}/{len(oe_scenarios)}) models"
            print(f"\n🚀 ===== Training OSR Model: {oe_tag} ===== [{current_progress}]")
            print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💾 GPU Memory before training: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            self.logger.info(
                f"[Start] Scenario={oe_tag}, type={scenario_type}, algo={scenario_algorithm}, progress={current_progress}, samples={scenario_sample_count}"
            )

            # Load tokenizer for Self-Attention functionality
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_NAME,
                cache_dir=self.config.HUGGINGFACE_CACHE_DIR
            )

            # Create model with scenario-specific Self-Attention setting
            model = SimpleOEModel(
                config=self.config,
                num_labels=num_classes,
                label2id=label2id,
                id2label=id2label,
                class_weights=None,
                tokenizer=tokenizer,
                use_self_attention_oe=enable_self_attention,
                oe_dataset_provided=oe_dataset is not None
            )

            # Hendrycks-style fresh initialization for Stage3
            # For Hendrycks simultaneous ID+OE training, we should start from fresh pretrained weights,
            # NOT from a model already trained on ID data (Standard checkpoint).
            # This ensures the model learns ID/OOD discrimination from scratch.
            if oe_tag != 'Standard' and self.standard_model_state is not None and not stage3_only_request:
                # Only load Standard weights if NOT in stage3-only mode
                try:
                    missing, unexpected = model.model.load_state_dict(self.standard_model_state, strict=False)
                    if missing:
                        print(f"⚠️  Missing keys when loading baseline weights for {oe_tag}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
                    if unexpected:
                        print(f"⚠️  Unexpected keys when loading baseline weights for {oe_tag}: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
                    else:
                        print(f"✅ Initialized {oe_tag} from cached Standard weights")
                except Exception as exc:
                    print(f"⚠️  Failed to load baseline weights for {oe_tag}: {exc}")
            elif oe_tag != 'Standard' and stage3_only_request:
                # Stage3-only: Use fresh pretrained RoBERTa (Hendrycks-style)
                print(f"🆕 {oe_tag}: Fresh initialization from pretrained {self.config.MODEL_NAME} (Hendrycks-style)")
            elif oe_tag == 'Standard':
                print(f"🆕 {oe_tag}: Fresh initialization from pretrained {self.config.MODEL_NAME}")
            
            # Setup data loader
            if oe_dataset:
                oe_batch_size = self.config.OSR_BATCH_SIZE
                oe_loader = DataLoader(
                    oe_dataset,
                    batch_size=oe_batch_size,
                    shuffle=True,
                    num_workers=self.config.OSR_NUM_DATALOADER_WORKERS
                )
                train_loader = CombinedOSRDataLoader(
                    id_train_loader,
                    oe_loader,
                    ratio=1
                )
            else:
                train_loader = id_train_loader
            
            # Setup trainer
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.OSR_EARLY_STOPPING_PATIENCE,
                    mode='min',
                    min_delta=self.config.OSR_EARLY_STOPPING_MIN_DELTA
                )
            ]

            if not getattr(self.config, 'DISABLE_MODEL_CHECKPOINT', False):
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=self.config.OSR_MODEL_DIR,
                        filename=f'osr_{oe_tag}_{dataset_name}',
                        monitor='val_loss',
                        mode='min',
                        save_top_k=1,
                        save_last=True,
                        verbose=True
                    )
                )

            # Add GPU monitoring if available
            if torch.cuda.is_available() and getattr(self.config, 'ENABLE_DEVICE_STATS_MONITOR', False):
                callbacks.append(DeviceStatsMonitor())

            # Create detailed config dict for WandB
            wandb_run_name = f"{dataset_name}_{oe_tag}_{'holdout' if self.config.OSR_EXPERIMENT_METHOD == 'holdout' else 'cross'}"

            osr_config = {
                'dataset': dataset_name,
                'oe_source': oe_tag,
                'oe_method': 'SelfAttention' if enable_self_attention else 'Traditional',
                'oe_method_config': self.config.OE_METHOD,
                'model_name': self.config.MODEL_NAME,
                'max_length': self.config.OSR_MAX_LENGTH,
                'batch_size': self.config.OSR_BATCH_SIZE,
                'learning_rate': self.config.OSR_LEARNING_RATE,
                'num_epochs': self.config.OSR_NUM_EPOCHS,
                'early_stopping_patience': self.config.OSR_EARLY_STOPPING_PATIENCE,
                'experiment_type': 'osr',
                'use_amp': True,
                'use_sdpa': True,
                'attention_filtering_method': scenario_algorithm if enable_self_attention else "N/A",
                'attention_top_k': self.config.ATTENTION_TOP_K if enable_self_attention else "N/A",
                'attention_top_p': self.config.ATTENTION_TOP_P if enable_self_attention else "N/A",
                'min_attention_top_tokens': self.config.MIN_ATTENTION_TOP_TOKENS if enable_self_attention else "N/A",
                'use_attention_elbow_refinement': self.config.USE_ATTENTION_ELBOW_REFINEMENT if enable_self_attention else "N/A",
                'masking_probability': self.config.MASKING_PROBABILITY if enable_self_attention else "N/A",
                'holdout_classes': holdout_info['class_names'] if holdout_info else []
            }

            # Create appropriate tags
            tags = ['osr', dataset_name, oe_tag, self.config.OE_METHOD]
            if enable_self_attention:
                tags.append('self_attention_oe')
            if self.config.USE_TRADITIONAL_OE:
                tags.append('traditional_oe')

            # Initialize pytorch_logger
            pytorch_logger = None
            try:
                pytorch_logger = setup_pytorch_logger(
                    log_dir=self.config.LOG_DIR,
                    name=wandb_run_name,
                    project=self.config.WANDB_PROJECT,  # Use the configured project name
                    config_dict=osr_config,
                    tags=tags
                )
            except Exception as e:
                print(f"Warning: Failed to setup pytorch_logger: {e}")
                pytorch_logger = None

            if pytorch_logger and hasattr(pytorch_logger, 'experiment'):
                scenario_metadata = {
                    'scenario/name': oe_tag,
                    'scenario/index': scenario_idx,
                    'scenario/total': len(oe_scenarios),
                    'scenario/type': scenario_type,
                    'scenario/algorithm': scenario_algorithm or 'N/A',
                    'scenario/oe_samples': scenario_sample_count,
                    'scenario/oe_limit': scenario.get('sample_limit', self.current_oe_sample_limit),
                    'scenario/holdout': self.config.OSR_EXPERIMENT_METHOD,
                    'scenario/holdout_classes': ', '.join(holdout_info['class_names']) if holdout_info else 'N/A',
                    'event': 'scenario_start'
                }
                if enable_self_attention:
                    scenario_metadata.update({
                        'params/attention_top_p': safe_float_for_wandb(self.config.ATTENTION_TOP_P),
                        'params/masking_probability': safe_float_for_wandb(self.config.MASKING_PROBABILITY),
                        'params/oe_uniform_loss_weight': safe_float_for_wandb(self.config.OE_UNIFORM_LOSS_WEIGHT)
                    })
                try:
                    pytorch_logger.experiment.log(scenario_metadata)
                except Exception as e:
                    print(f"⚠️  Failed to log scenario metadata to WandB: {e}")
                if pytorch_logger and hasattr(pytorch_logger, 'experiment'):
                    if torch.cuda.is_available():
                        try:
                            free_bytes, total_bytes = torch.cuda.mem_get_info()
                            used_bytes = total_bytes - free_bytes
                            utilization = (used_bytes / total_bytes * 100) if total_bytes else 0.0
                            vram_payload = {
                                'scenario/name': oe_tag,
                                'event': 'vram_start',
                                'system/VRAM_allocated_MB_start': torch.cuda.memory_allocated() / 1024**2,
                                'system/VRAM_reserved_MB_start': torch.cuda.memory_reserved() / 1024**2,
                                'system/VRAM_free_MB_start': free_bytes / 1024**2,
                                'system/VRAM_total_MB_start': total_bytes / 1024**2,
                                'system/VRAM_used_MB_start': used_bytes / 1024**2,
                                'system/VRAM_utilization_pct_start': utilization
                            }
                            pytorch_logger.experiment.log(vram_payload)
                        except Exception as e:
                            print(f"⚠️  Failed to log VRAM stats: {e}")
                            fallback_payload = {
                                'scenario/name': oe_tag,
                                'event': 'vram_start',
                                'system/VRAM_allocated_MB_start': 0.0,
                                'system/VRAM_reserved_MB_start': 0.0,
                                'system/VRAM_free_MB_start': 0.0,
                                'system/VRAM_total_MB_start': 0.0,
                                'system/VRAM_used_MB_start': 0.0,
                                'system/VRAM_utilization_pct_start': 0.0
                            }
                            pytorch_logger.experiment.log(fallback_payload)
                    else:
                        fallback_payload = {
                            'scenario/name': oe_tag,
                            'event': 'vram_start',
                            'system/VRAM_allocated_MB_start': 0.0,
                            'system/VRAM_reserved_MB_start': 0.0,
                            'system/VRAM_free_MB_start': 0.0,
                            'system/VRAM_total_MB_start': 0.0,
                            'system/VRAM_used_MB_start': 0.0,
                            'system/VRAM_utilization_pct_start': 0.0
                        }
                        pytorch_logger.experiment.log(fallback_payload)
            
            trainer = pl.Trainer(
                max_epochs=self.config.OSR_NUM_EPOCHS,
                callbacks=callbacks,
                logger=pytorch_logger,
                accelerator='auto',
                devices='auto',
                deterministic=True,
                precision='16-mixed',  # Enable AMP (Automatic Mixed Precision)
                enable_progress_bar=False,  # Disable progress bar for clean logs
                log_every_n_steps=10  # More frequent logging for better monitoring
            )
            
            # Train
            trainer.fit(model, train_loader, self.data_module.val_dataloader())

            callback_metrics = trainer.callback_metrics if trainer else {}
            id_val_acc = tensor_to_float(callback_metrics.get('val_acc'))
            id_val_f1_macro = tensor_to_float(callback_metrics.get('val_f1_macro'))
            id_val_f1_micro = tensor_to_float(callback_metrics.get('val_f1_micro'))
            id_val_precision = tensor_to_float(callback_metrics.get('val_precision'))
            id_val_recall = tensor_to_float(callback_metrics.get('val_recall'))
            
            # Evaluate on OOD datasets
            print(f"\n===== Evaluating Model: {oe_tag} =====")
            
            device = next(model.parameters()).device
            
            # Calculate ID scores
            id_scores = calculate_msp_scores(model, id_test_loader, device)
            
            scenario_metrics = []

            # Evaluate on each OOD dataset
            for ood_name, ood_dataset in ood_datasets.items():
                # Create OOD data loader using OSRTextDataset
                ood_loader = DataLoader(
                    ood_dataset,
                    batch_size=self.config.EVAL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.config.NUM_WORKERS
                )

                ood_scores = calculate_msp_scores(model, ood_loader, device)
                metrics = get_ood_metrics(id_scores, ood_scores)

                scenario_metrics.append({
                    'AUROC': metrics['AUROC'],
                    'AUPR': metrics['AUPR'],
                    'FPR95': metrics['FPR95'],
                    'ood_name': ood_name
                })

                # Log OOD metrics to WandB if available
                if pytorch_logger and hasattr(pytorch_logger, 'experiment'):
                    metric_payload = {
                        'scenario/name': oe_tag,
                        'scenario/index': scenario_idx,
                        'scenario/type': scenario_type,
                        'scenario/algorithm': scenario_algorithm or 'N/A',
                        'scenario/oe_samples': scenario_sample_count,
                        'scenario/oe_limit': scenario.get('sample_limit', self.current_oe_sample_limit),
                        'counts/OE_total_input': scenario_total_input,
                        'counts/ProcessedOE': scenario_processed_count,
                        'counts/GeneratedOE': scenario_sample_count,
                        'ood/name': ood_name,
                        'counts/ID': len(id_scores),
                        'counts/OOD': len(ood_scores),
                        'event': 'ood_metrics'
                    }
                    metric_payload.update({
                        'id/val_acc': safe_float_for_wandb(id_val_acc),
                        'id/val_f1_macro': safe_float_for_wandb(id_val_f1_macro),
                        'id/val_f1_micro': safe_float_for_wandb(id_val_f1_micro),
                        'id/val_precision': safe_float_for_wandb(id_val_precision),
                        'id/val_recall': safe_float_for_wandb(id_val_recall)
                    })
                    if enable_self_attention:
                        metric_payload.update({
                            'params/attention_top_p': safe_float_for_wandb(self.config.ATTENTION_TOP_P),
                            'params/masking_probability': safe_float_for_wandb(self.config.MASKING_PROBABILITY),
                            'params/oe_uniform_loss_weight': safe_float_for_wandb(self.config.OE_UNIFORM_LOSS_WEIGHT)
                        })
                    metric_payload[format_wandb_metric_key('ood', ood_name, 'AUROC')] = metrics['AUROC']
                    metric_payload[format_wandb_metric_key('ood', ood_name, 'AUPR')] = metrics['AUPR']
                    metric_payload[format_wandb_metric_key('ood', ood_name, 'FPR95')] = metrics['FPR95']
                    try:
                        pytorch_logger.experiment.log(metric_payload)
                    except Exception as e:
                        print(f"⚠️  Failed to log OOD metrics to WandB: {e}")

                # Determine OE method based on current config
                if self.config.USE_SELF_ATTENTION_OE and self.config.USE_TRADITIONAL_OE:
                    oe_method = "Both"
                elif self.config.USE_SELF_ATTENTION_OE:
                    oe_method = "SelfAttention"
                elif self.config.USE_TRADITIONAL_OE:
                    oe_method = "Traditional"
                else:
                    oe_method = "Baseline"
                result = {
                    'ID_Dataset': dataset_name,
                    'OE_Source': oe_tag,
                    'OE_Method_Type': oe_method,
                    'OE_Method_Config': self.config.OE_METHOD,
                    'OOD_Dataset': ood_name,
                    'AUROC': metrics['AUROC'],
                    'AUPR': metrics['AUPR'],
                    'FPR95': metrics['FPR95'],
                    'ID_Count': len(id_scores),
                    'OOD_Count': len(ood_scores),
                    'Attention_Filtering_Method': scenario_algorithm if enable_self_attention else "N/A",
                    'Attention_Top_K': self.config.ATTENTION_TOP_K if enable_self_attention else np.nan,
                    'Attention_Top_P': float(self.config.ATTENTION_TOP_P) if enable_self_attention else np.nan,
                    'Masking_Probability': float(self.config.MASKING_PROBABILITY) if enable_self_attention else np.nan,
                    'Hard_Negative_Ratio': np.nan,
                    'Holdout_Classes': ', '.join(holdout_info['class_names']) if holdout_info else "N/A"
                }
                result['OE_Total_Input'] = scenario_total_input
                result['OE_Processed_Count'] = scenario_processed_count
                result['OE_Generated_Count'] = scenario_sample_count
                result['OE_Generation_Ratio'] = (
                    scenario_sample_count / scenario_processed_count
                    if scenario_processed_count
                    else 0.0
                )
                result['ID_Val_Acc'] = id_val_acc
                result['ID_Val_F1_Macro'] = id_val_f1_macro
                result['ID_Val_F1_Micro'] = id_val_f1_micro
                result['ID_Val_Precision'] = id_val_precision
                result['ID_Val_Recall'] = id_val_recall
                results.append(result)
                
                print(f"  {ood_name}: AUROC={metrics['AUROC']:.4f}, "
                      f"AUPR={metrics['AUPR']:.4f}, FPR95={metrics['FPR95']:.4f}")

            # Finalize WandB run for this OSR scenario
            # WandB finalization with proper error handling
            if pytorch_logger and hasattr(pytorch_logger, 'experiment'):
                try:
                    valid_aurocs = [m['AUROC'] for m in scenario_metrics if not np.isnan(m['AUROC'])]
                    valid_auprs = [m['AUPR'] for m in scenario_metrics if not np.isnan(m['AUPR'])]
                    valid_fpr = [m['FPR95'] for m in scenario_metrics if not np.isnan(m['FPR95'])]

                    if WANDB_AVAILABLE and wandb is not None and scenario_metrics:
                        try:
                            metrics_table = wandb.Table(columns=['OOD_Dataset', 'AUROC', 'AUPR', 'FPR95'])
                            for row in scenario_metrics:
                                metrics_table.add_data(
                                    row['ood_name'],
                                    safe_float_for_wandb(row['AUROC']),
                                    safe_float_for_wandb(row['AUPR']),
                                    safe_float_for_wandb(row['FPR95'])
                                )

                            pytorch_logger.experiment.log({'tables/ood_metrics': metrics_table})

                            for metric_key in ['AUROC', 'AUPR', 'FPR95']:
                                try:
                                    bar_plot = wandb.plot.bar(
                                        metrics_table,
                                        'OOD_Dataset',
                                        metric_key,
                                        title=f"{oe_tag} {metric_key} by OOD dataset"
                                    )
                                    pytorch_logger.experiment.log({f'plots/{metric_key}_by_ood': bar_plot})
                                except Exception as chart_error:
                                    print(f"⚠️  Failed to log WandB {metric_key} plot: {chart_error}")
                        except Exception as table_error:
                            print(f"⚠️  Failed to build WandB OOD metrics table: {table_error}")

                    summary_payload = {
                        'scenario/name': oe_tag,
                        'scenario/index': scenario_idx,
                        'scenario/total': len(oe_scenarios),
                        'scenario/type': scenario_type,
                        'scenario/algorithm': scenario_algorithm or 'N/A',
                        'scenario/oe_samples': scenario_sample_count,
                        'counts/OE_total_input': scenario_total_input,
                        'counts/ProcessedOE': scenario_processed_count,
                        'counts/GeneratedOE': scenario_sample_count,
                        'scenario/oe_limit': scenario.get('sample_limit', self.current_oe_sample_limit),
                        'scenario/holdout': self.config.OSR_EXPERIMENT_METHOD,
                        'scenario/holdout_classes': ', '.join(holdout_info['class_names']) if holdout_info else 'N/A',
                        'scenario/ood_evaluated': len(scenario_metrics),
                        'summary/AUROC_mean': float(np.mean(valid_aurocs)) if valid_aurocs else float('nan'),
                        'summary/AUPR_mean': float(np.mean(valid_auprs)) if valid_auprs else float('nan'),
                        'summary/FPR95_mean': float(np.mean(valid_fpr)) if valid_fpr else float('nan'),
                        'event': 'scenario_complete'
                    }
                    summary_payload.update({
                        'id/val_acc': safe_float_for_wandb(id_val_acc),
                        'id/val_f1_macro': safe_float_for_wandb(id_val_f1_macro),
                        'id/val_f1_micro': safe_float_for_wandb(id_val_f1_micro),
                        'id/val_precision': safe_float_for_wandb(id_val_precision),
                        'id/val_recall': safe_float_for_wandb(id_val_recall)
                    })
                    if enable_self_attention:
                        summary_payload.update({
                            'params/attention_top_p': safe_float_for_wandb(self.config.ATTENTION_TOP_P),
                            'params/masking_probability': safe_float_for_wandb(self.config.MASKING_PROBABILITY),
                            'params/oe_uniform_loss_weight': safe_float_for_wandb(self.config.OE_UNIFORM_LOSS_WEIGHT)
                        })
                    for row in scenario_metrics:
                        dataset_key = row['ood_name']
                        summary_payload[format_wandb_metric_key('summary', dataset_key, 'AUROC')] = safe_float_for_wandb(row['AUROC'])
                        summary_payload[format_wandb_metric_key('summary', dataset_key, 'AUPR')] = safe_float_for_wandb(row['AUPR'])
                        summary_payload[format_wandb_metric_key('summary', dataset_key, 'FPR95')] = safe_float_for_wandb(row['FPR95'])
                    if torch.cuda.is_available():
                        try:
                            free_bytes_end, total_bytes_end = torch.cuda.mem_get_info()
                            used_bytes_end = total_bytes_end - free_bytes_end
                            utilization_end = (used_bytes_end / total_bytes_end * 100) if total_bytes_end else 0.0
                            summary_payload.update({
                                'system/VRAM_allocated_MB_end': torch.cuda.memory_allocated() / 1024**2,
                                'system/VRAM_reserved_MB_end': torch.cuda.memory_reserved() / 1024**2,
                                'system/VRAM_free_MB_end': free_bytes_end / 1024**2,
                                'system/VRAM_total_MB_end': total_bytes_end / 1024**2,
                                'system/VRAM_used_MB_end': used_bytes_end / 1024**2,
                                'system/VRAM_utilization_pct_end': utilization_end
                            })
                        except Exception as e:
                            print(f"⚠️  Failed to log ending VRAM stats: {e}")
                            summary_payload.update({
                                'system/VRAM_allocated_MB_end': 0.0,
                                'system/VRAM_reserved_MB_end': 0.0,
                                'system/VRAM_free_MB_end': 0.0,
                                'system/VRAM_total_MB_end': 0.0,
                                'system/VRAM_used_MB_end': 0.0,
                                'system/VRAM_utilization_pct_end': 0.0
                            })
                    else:
                        summary_payload.update({
                            'system/VRAM_allocated_MB_end': 0.0,
                            'system/VRAM_reserved_MB_end': 0.0,
                            'system/VRAM_free_MB_end': 0.0,
                            'system/VRAM_total_MB_end': 0.0,
                            'system/VRAM_used_MB_end': 0.0,
                            'system/VRAM_utilization_pct_end': 0.0
                        })
                    pytorch_logger.experiment.log(summary_payload)
                    pytorch_logger.experiment.finish()
                    print(f"📊 WandB run completed for {oe_tag}")
                except Exception as e:
                    print(f"⚠️  Failed to finalize WandB run: {e}")

            if oe_tag == 'Standard':
                self.standard_model_state = {k: v.detach().cpu() for k, v in model.model.state_dict().items()}

            # Generate Self-Attention OE datasets after Standard training using trained model
            if (oe_tag == 'Standard'
                    and self.config.OE_METHOD in ['self_attention', 'both']
                    and any(s.get('needs_generation') and s.get('dataset') is None for s in oe_scenarios)):

                print("\n🧠 Generating Self-Attention OE datasets from trained Standard model...")
                standard_state = dict(self.standard_model_state)

                # Persist Standard checkpoint for future staged stage3 reuse
                if self.config.ATTENTION_GENERATION_MODE == 'staged':
                    checkpoint_dir = self._stage3_checkpoint_path.parent
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(standard_state, self._stage3_checkpoint_path)
                    self.config.STAGE3_STANDARD_CHECKPOINT = str(self._stage3_checkpoint_path)
                    print(f"   💾 Saved Standard model state to {self._stage3_checkpoint_path}")

                for pending in oe_scenarios:
                    if not pending.get('needs_generation') or pending.get('dataset') is not None:
                        continue

                    algo = pending.get('algorithm')
                    if not algo or algo in generated_self_attention_datasets:
                        continue

                    print(f"   ▶ Generating dataset for algorithm: {algo}")
                    generated_ds, generation_stats = self._generate_attention_based_oe_data(
                        id_train_samples,
                        tokenizer,
                        algo,
                        model_state_dict=standard_state,
                        num_labels=self.data_module.num_classes,
                        sample_limit=getattr(self, 'current_oe_sample_limit', None)
                    )

                    pending['oe_generation_stats'] = generation_stats

                    if generated_ds and len(generated_ds) > 0:
                        generated_self_attention_datasets[algo] = generated_ds
                        generated_self_attention_stats[algo] = generation_stats
                        pending['dataset'] = generated_ds
                        pending['oe_sample_count'] = len(generated_ds)
                        pending['oe_processed_count'] = generation_stats.get('processed_samples', len(generated_ds))
                        pending['oe_total_input'] = generation_stats.get('total_input_samples', len(id_train_samples))
                        print(f"   ✅ Generated {len(generated_ds)} samples for {algo}")
                    else:
                        print(f"   ⚠️ No OE samples generated for {algo}")

            # Explicit memory cleanup after each model
            print(f"🧹 Cleaning up memory for {oe_tag}...")
            print(f"💾 GPU Memory before cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

            # Delete model and trainer references
            del model
            del trainer
            if 'oe_loader' in locals():
                del oe_loader
            if 'train_loader' in locals() and train_loader != id_train_loader:
                del train_loader

            # Force garbage collection
            import gc
            gc.collect()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print(f"💾 GPU Memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"✅ Model {oe_tag} training and evaluation completed successfully!")
            self.logger.info(
                f"[Complete] Scenario={oe_tag}, type={scenario_type}, algo={scenario_algorithm}, OOD evaluated={len(scenario_metrics)}"
            )

            # Wait a moment to ensure cleanup
            import time
            time.sleep(2)

            self.config.ATTENTION_FILTERING_METHOD = previous_filter_method

            if stage2_only_request and oe_tag == 'Standard':
                print("\n⏹️  Stage2 extraction requested only; skipping remaining OE scenarios.")
                self.logger.info("Stage2-only run detected – terminating after Standard model and attention export.")
                break
        
        # Save results with error handling
        try:
            results_df = pd.DataFrame(results)
            os.makedirs(self.config.RESULTS_DIR, exist_ok=True)

            # Append run timestamp for traceability
            run_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            results_df["Run_Timestamp"] = run_timestamp

            results_path = os.path.join(self.config.RESULTS_DIR, f'osr_results_{dataset_name}.csv')
            if os.path.exists(results_path):
                existing_df = pd.read_csv(results_path)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_csv(results_path, index=False)
            self.logger.info(f"✅ Results saved to: {results_path}")

            summary_output_path = os.path.join(
                self.config.RESULTS_DIR,
                f'osr_results_{dataset_name}_self_attention_summary.csv'
            )
            self_attention_df = results_df.dropna(subset=['Attention_Top_P', 'Masking_Probability'], how='any')
            if not self_attention_df.empty:
                grouping_columns = [
                    'Attention_Filtering_Method',
                    'Attention_Top_P',
                    'Masking_Probability',
                    'Hard_Negative_Ratio',
                    'OOD_Dataset'
                ]
                aggregation_spec = {
                    'AUROC': 'mean',
                    'AUPR': 'mean',
                    'FPR95': 'mean',
                    'ID_Val_F1_Macro': 'mean',
                    'ID_Val_F1_Micro': 'mean',
                    'ID_Val_Acc': 'mean',
                    'Run_Timestamp': 'first'
                }
                grouped = self_attention_df.groupby(grouping_columns, dropna=False)
                summary_df = grouped.agg(aggregation_spec)
                summary_df['Run_Count'] = grouped.size()
                summary_df = summary_df.reset_index().sort_values(grouping_columns)
                summary_df.to_csv(summary_output_path, index=False)
                self.logger.info(
                    f"✅ Self-attention summary saved to: {summary_output_path}"
                )
            else:
                self.logger.info("No self-attention rows found; skipping summary CSV generation.")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save CSV results: {e}")
            print(f"❌ Failed to save CSV results: {e}")
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"SUMMARY FOR {dataset_name}")
        print(f"{'='*50}")
        
        for ood_name in results_df['OOD_Dataset'].unique():
            ood_results = results_df[results_df['OOD_Dataset'] == ood_name]
            print(f"\nOOD Dataset: {ood_name}")
            for _, row in ood_results.iterrows():
                print(f"  {row['OE_Source']:20} → AUROC={row['AUROC']:.4f}")
        
        return results_df

# =============================================================================
# Main Function and CLI
# =============================================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simplified Outlier Exposure Implementation')
    
    # Basic arguments
    parser.add_argument('--dataset', type=str, default='ag_news',
                       choices=list(Config.NLP_DATASETS.keys()),
                       help='Dataset to use for experiments')
    parser.add_argument('--mode', type=str, default='baseline',
                       choices=['baseline', 'traditional_oe', 'self_attention_oe', 'both_oe'],
                       help='Experiment mode: baseline (no OE), traditional_oe (external datasets), self_attention_oe (attention-guided), or both_oe (compare both methods)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='simplified_oe_experiments',
                       help='Output directory for results')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='roberta-base',
                       help='Pretrained model name')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--osr_experiment_method', type=str, default='cross',
                       choices=['cross', 'holdout'], help='OSR evaluation method')
    parser.add_argument('--holdout_ratio', type=float, default=0.2,
                       help='Fraction of classes to hold out for OOD evaluation')
    parser.add_argument('--holdout_min_classes', type=int, default=1,
                       help='Minimum number of classes to hold out')
    parser.add_argument('--attention_generation_max_samples', type=int, default=-1,
                       help='Maximum samples to use when generating Self-Attention OE data (<=0 uses OE limit)')
    parser.add_argument('--oe_max_samples', type=int, default=-1,
                       help='Maximum OE samples per scenario (<=0 uses ID train sample count)')
    parser.add_argument('--attention_top_p', type=float, default=None,
                       help='Cumulative ratio (0-1) for selecting high-attention tokens before elbow refinement')
    parser.add_argument('--min_attention_top_tokens', type=int, default=None,
                       help='Minimum number of tokens to keep after attention filtering')
    parser.add_argument('--disable_attention_elbow_refinement', action='store_true',
                       help='Skip elbow refinement after top-p selection for attention filtering')
    parser.add_argument('--enable_attention_debug_logs', action='store_true',
                       help='Emit detailed attention selection debug logs (limited volume)')
    parser.add_argument('--attention_debug_max_examples', type=int, default=None,
                       help='Maximum debug log entries per attention category (default=5)')
    parser.add_argument('--disable_model_checkpoint', action='store_true',
                       help='Disable saving lightning model checkpoints (useful for sweeps)')
    parser.add_argument('--masking_probability', type=float, default=None,
                       help='Override probability for masking selected tokens (default=0.3)')
    parser.add_argument('--self_attention_loss_weight', type=float, default=None,
                       help='Override loss weight for self-attention hard negatives (default=0.5)')
    parser.add_argument('--oe_uniform_loss_weight', type=float, default=None,
                       help='Override weight applied to uniform OE loss term (default=1.0)')
    parser.add_argument('--attention_generation_mode', type=str, default=None,
                       choices=['on_the_fly', 'staged'],
                       help='Select self-attention OE generation mode (on-the-fly or staged two-phase)')
    parser.add_argument('--attention_stage', type=str, default=None,
                       choices=['both', 'stage2', 'stage3'],
                       help='When using staged generation, choose which stage(s) to run in this invocation')
    parser.add_argument('--attention_cache_dir', type=str, default=None,
                       help='Directory for caching staged attention scores and metadata')
    parser.add_argument('--attention_stage2_shard_size', type=int, default=None,
                       help='Number of samples per Stage2 shard file when caching attention scores')
    # --disable_attention_stage3_masking removed: masking is always enabled (Hendrycks requirement)
    parser.add_argument('--stage3_use_cached_standard_model', action='store_true',
                       help='When running staged stage3 only, reuse a saved Standard checkpoint and skip retraining')
    parser.add_argument('--stage3_standard_checkpoint', type=str, default=None,
                       help='Optional path to the saved Standard model state for staged stage3 reuse')
    parser.add_argument('--enable_device_stats_monitor', action='store_true',
                       help='Log detailed GPU allocator metrics via DeviceStatsMonitor (off by default)')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    
    # Update config with CLI arguments
    Config.CURRENT_NLP_DATASET = args.dataset
    Config.OUTPUT_DIR = args.output_dir
    Config.MODEL_NAME = args.model_name
    Config.MAX_LENGTH = args.max_length
    Config.BATCH_SIZE = args.batch_size
    Config.EVAL_BATCH_SIZE = args.batch_size
    Config.OSR_BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.learning_rate
    Config.NUM_EPOCHS = args.num_epochs
    Config.OSR_NUM_EPOCHS = args.num_epochs  # OSR epochs should match ID epochs for fair comparison
    Config.RANDOM_STATE = args.seed
    Config.OSR_EXPERIMENT_METHOD = args.osr_experiment_method
    Config.OSR_HOLDOUT_RATIO = args.holdout_ratio
    Config.OSR_HOLDOUT_MIN_CLASSES = args.holdout_min_classes
    Config.ATTENTION_GENERATION_MAX_SAMPLES = None if args.attention_generation_max_samples <= 0 else args.attention_generation_max_samples
    Config.OSR_LEARNING_RATE = args.learning_rate
    Config.OE_MAX_SAMPLES = None if args.oe_max_samples <= 0 else args.oe_max_samples
    Config.OE_SAMPLING_SEED = args.seed

    if args.attention_top_p is not None:
        Config.ATTENTION_TOP_P = max(0.0, min(1.0, args.attention_top_p))
    if args.min_attention_top_tokens is not None and args.min_attention_top_tokens > 0:
        Config.MIN_ATTENTION_TOP_TOKENS = args.min_attention_top_tokens
    if args.disable_attention_elbow_refinement:
        Config.USE_ATTENTION_ELBOW_REFINEMENT = False
    if args.enable_attention_debug_logs:
        Config.ENABLE_ATTENTION_DEBUG_LOGS = True
    if args.attention_debug_max_examples is not None and args.attention_debug_max_examples > 0:
        Config.ATTENTION_DEBUG_MAX_EXAMPLES = args.attention_debug_max_examples
    if args.disable_model_checkpoint:
        Config.DISABLE_MODEL_CHECKPOINT = True
    if args.masking_probability is not None:
        Config.MASKING_PROBABILITY = min(max(0.0, float(args.masking_probability)), 1.0)
    if args.self_attention_loss_weight is not None:
        Config.SELF_ATTENTION_LOSS_WEIGHT = max(0.0, float(args.self_attention_loss_weight))
    if args.oe_uniform_loss_weight is not None:
        Config.OE_UNIFORM_LOSS_WEIGHT = max(0.0, float(args.oe_uniform_loss_weight))
    if args.attention_generation_mode is not None:
        Config.ATTENTION_GENERATION_MODE = args.attention_generation_mode
    if args.attention_stage is not None:
        Config.ATTENTION_STAGE_TO_RUN = args.attention_stage
    if args.attention_cache_dir:
        Config.ATTENTION_CACHE_DIR = args.attention_cache_dir
    if args.attention_stage2_shard_size is not None and args.attention_stage2_shard_size > 0:
        Config.ATTENTION_STAGE2_SHARD_SIZE = args.attention_stage2_shard_size
    # disable_attention_stage3_masking removed: masking is always enabled
    if args.stage3_use_cached_standard_model:
        Config.STAGE3_USE_CACHED_STANDARD_MODEL = True
    if args.stage3_standard_checkpoint:
        Config.STAGE3_STANDARD_CHECKPOINT = args.stage3_standard_checkpoint
    Config.ENABLE_DEVICE_STATS_MONITOR = bool(args.enable_device_stats_monitor)

    auto_adjust_lr = np.isclose(args.learning_rate, Config.BASE_LEARNING_RATE)
    Config.adjust_learning_rates_for_batch_size(auto_adjust=auto_adjust_lr)
    
    Config.create_directories()
    
    # Set OE mode based on experiment mode
    Config.set_oe_mode(args.mode)
    
    # Setup logging
    logger = setup_logging(Config.OUTPUT_DIR, args.dataset, args.mode)

    # Get timestamp for progress logging
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create pipeline
    pipeline = SimplifiedOEPipeline(Config)
    
    logger.info(f"🚀 Starting Simplified OE Experiments")
    logger.info(f"   Dataset: {args.dataset}")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   Model: {args.model_name}")
    logger.info(f"   Seed: {args.seed}")
    logger.info(f"   Output Directory: {Config.OUTPUT_DIR}")
    logger.info(f"   Configuration: {Config.__dict__}")
    
    # Setup progress log file for real-time monitoring
    progress_log_path = os.path.join(Config.LOG_DIR, f"progress_{args.mode}_{args.dataset}_{timestamp}.log")

    # Ensure both stdout and file logging
    class DualLogger:
        def __init__(self, log_path):
            self.log_path = log_path

        def write_progress(self, message):
            print(message, flush=True)  # Console output
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
                f.flush()

    dual_logger = DualLogger(progress_log_path)
    dual_logger.write_progress(f"🚀 실시간 진행 상황 로그 시작 - {timestamp}")
    dual_logger.write_progress(f"📁 로그 파일: {progress_log_path}")

    results_df = None

    try:
        # Run experiments based on mode
        if args.mode == 'baseline':
            logger.info(f"\n🎯 Running baseline training (No OE)...")
            dual_logger.write_progress("🎯 Baseline 훈련 시작...")
            pipeline.run_baseline_training(args.dataset)
            logger.info("✅ Baseline training completed successfully")
            dual_logger.write_progress("✅ Baseline 훈련 완료")
            
        elif args.mode == 'traditional_oe':
            logger.info(f"\n📊 Running Traditional OE experiments...")
            dual_logger.write_progress("📊 Traditional OE 실험 시작...")
            Config.set_oe_mode('traditional_oe')  # 🔧 모드 설정 추가
            pipeline = SimplifiedOEPipeline(Config)  # 설정 변경 후 파이프라인 재생성
            results_df = pipeline.run_osr_experiments(args.dataset)
            dual_logger.write_progress("✅ Traditional OE 실험 완료")

        elif args.mode == 'self_attention_oe':
            logger.info(f"\n🧠 Running Self Attention-guided OE experiments...")
            dual_logger.write_progress("🧠 Self Attention-guided OE 실험 시작...")
            Config.set_oe_mode('self_attention_oe')  # 🔧 핵심 수정: 모드 설정 추가
            pipeline = SimplifiedOEPipeline(Config)  # 설정 변경 후 파이프라인 재생성
            results_df = pipeline.run_osr_experiments(args.dataset)
            dual_logger.write_progress("✅ Self Attention-guided OE 실험 완료")
            
        elif args.mode == 'both_oe':
            logger.info(f"\n🔄 Running Both OE methods comparison...")
            
            # Run Traditional OE first
            logger.info(f"\n📊 Phase 1: Traditional OE...")
            Config.set_oe_mode('traditional_oe')
            pipeline_traditional = SimplifiedOEPipeline(Config)
            results_traditional = pipeline_traditional.run_osr_experiments(args.dataset)
            
            # Run Self Attention-guided OE second  
            logger.info(f"\n🧠 Phase 2: Self Attention-guided OE...")
            Config.set_oe_mode('self_attention_oe')
            pipeline_self_attn = SimplifiedOEPipeline(Config)
            results_self_attn = pipeline_self_attn.run_osr_experiments(args.dataset)
            
            # Combine results for comparison
            results_df = pd.concat([results_traditional, results_self_attn], ignore_index=True)
            
        if args.mode in ['traditional_oe', 'self_attention_oe', 'both_oe']:
            if results_df is None:
                results_df = pipeline.run_osr_experiments(args.dataset)

            # Print best OE source for each OOD dataset (with NaN handling)
            logger.info(f"\n🏆 BEST OE SOURCES:")
            for ood_name in results_df['OOD_Dataset'].unique():
                ood_results = results_df[results_df['OOD_Dataset'] == ood_name]
                # Filter out NaN values before finding max
                valid_results = ood_results.dropna(subset=['AUROC'])
                if len(valid_results) > 0:
                    best_row = valid_results.loc[valid_results['AUROC'].idxmax()]
                    logger.info(f"  {ood_name:15} → {best_row['OE_Source']:20} (AUROC={best_row['AUROC']:.4f})")
                else:
                    logger.info(f"  {ood_name:15} → No valid results (all AUROC=NaN)")
            logger.info("✅ OSR experiments completed successfully")
        
        logger.info(f"\n✅ All experiments completed successfully!")
        logger.info(f"   Results saved in: {Config.OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
