"""
Enhanced Outlier Exposure (OE) Pipeline with OSR Experiments
Fixed version with consistent file paths and improved error handling
Supports both Syslog and NLP datasets with comprehensive experiments

[IMPROVEMENTS in this version]
- PyTorch Lightning integration for OSR training
- Removed GRU/LSTM components (focusing on Transformer models)
- Syslog "unknown" class as OOD dataset
- Pivot table generation for better result analysis
- Fixed removed_avg_attention in OE extraction
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset, random_split, ConcatDataset, WeightedRandomSampler
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
    GPT2Tokenizer
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, minmax_scale

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# 시각화 라이브러리
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("Warning: Seaborn not available. Some plots might not be generated.")

# 텍스트 처리
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from collections import defaultdict
import json
import math
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm import tqdm
import gc
from scipy.stats import entropy
import ast
from datetime import datetime
import random
import warnings
import logging

# HuggingFace 경고 메시지 숨기기
warnings.filterwarnings("ignore", message=".*Repo card metadata block.*")

# NLTK 초기화
NLTK_DATA_PATH = os.path.expanduser('~/AppData/Roaming/nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

# 전역 플래그로 한 번만 다운로드하도록 제어
_NLTK_DOWNLOADS_DONE = False

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def ensure_nltk_data():
    """NLTK 데이터가 있는지 확인하고, 필요시 다운로드"""
    global _NLTK_DOWNLOADS_DONE

    if _NLTK_DOWNLOADS_DONE:
        return

    required_data = [
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('stopwords', 'corpora/stopwords')
    ]

    for item_name, item_path in required_data:
        try:
            nltk.data.find(item_path)
            print(f"✅ {item_name} already available")
        except LookupError:
            print(f"📥 Downloading NLTK {item_name}...")
            try:
                nltk.download(item_name, quiet=True)
                print(f"✅ {item_name} downloaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to download {item_name}: {e}")
                if item_name == 'punkt_tab':
                    try:
                        print("📥 Trying fallback punkt download...")
                        nltk.download('punkt', quiet=True)
                        print("✅ punkt (fallback) downloaded successfully")
                    except Exception as fallback_e:
                        print(f"❌ Fallback punkt download also failed: {fallback_e}")

    _NLTK_DOWNLOADS_DONE = True

# 초기 다운로드 실행
ensure_nltk_data()

# === Device 처리 ===
def get_device():
    """안전한 device 획득"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_osr_device():
    """OSR용 device 획득"""
    return get_device()

# === 안전한 파일/디렉토리 처리 ===
def ensure_directory(path):
    """디렉토리 안전하게 생성"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        print(f"Warning: Could not create directory {path}: {e}")
        return False

def safe_load_dataset_file(file_path, fallback_encoding='utf-8'):
    """안전한 파일 로딩"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, FileNotFoundError) as e:
            if encoding == encodings[-1]:
                raise e
            continue
    
    return None

# === 안전한 DataLoader 생성 ===
def create_safe_dataloader(dataset, batch_size, shuffle=False, num_workers=None, collate_fn=None):
    """안전한 DataLoader 생성"""
    if num_workers is None:
        num_workers = min(2, os.cpu_count() or 1)
    
    # Windows에서 multiprocessing 문제 방지
    if os.name == 'nt':
        num_workers = 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0 and os.name != 'nt'
    )

# === Enhanced Configuration Class ===
class Config:
    """Enhanced Configuration Class - Supporting both Syslog and NLP datasets"""

    # === 기본 모드 선택 ===
    EXPERIMENT_MODE = "syslog"  # "syslog" 또는 "nlp"

    # === NLP Dataset 설정 ===
    NLP_DATASETS = {
        '20newsgroups': {'name': 'SetFit/20_newsgroups', 'subset': None, 'text_column': 'text', 'label_column': 'label'},
        'trec': {'name': 'trec', 'subset': None, 'text_column': 'text', 'label_column': 'coarse_label'},
        'sst2': {'name': 'glue', 'subset': 'sst2', 'text_column': 'sentence', 'label_column': 'label'},
        'wikitext': {'name': 'wikitext', 'subset': 'wikitext-2-raw-v1', 'text_column': 'text', 'label_column': None},
        'wmt16': {'name': 'wmt16', 'subset': 'ro-en', 'text_column': 'en', 'label_column': None}
    }

    # === NLP 특화 설정 ===
    CURRENT_NLP_DATASET = '20newsgroups'
    COMPREHENSIVE_NLP_EXPERIMENTS = False
    NLP_DATASETS_FOR_COMPREHENSIVE = ['20newsgroups', 'trec', 'sst2']

    # === 기존 Syslog 설정 ===
    ORIGINAL_DATA_PATH = 'data/log_all.csv'
    TEXT_COLUMN = 'text'
    CLASS_COLUMN = 'class'
    EXCLUDE_CLASS_FOR_TRAINING = "unknown"

    # === 출력 디렉토리 설정 ===
    OUTPUT_DIR = 'enhanced_oe_results_comprehensive'
    BASE_OUTPUT_DIR = OUTPUT_DIR  # 추가: 기본 출력 디렉토리 저장
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "base_classifier_model")
    LOG_DIR = os.path.join(OUTPUT_DIR, "lightning_logs")
    CONFUSION_MATRIX_DIR = os.path.join(LOG_DIR, "confusion_matrices")
    VIS_DIR = os.path.join(OUTPUT_DIR, "oe_extraction_visualizations")
    OE_DATA_DIR = os.path.join(OUTPUT_DIR, "extracted_oe_datasets")
    ATTENTION_DATA_DIR = os.path.join(OUTPUT_DIR, "attention_analysis")

    # === 모델 및 학습 공통 설정 ===
    MODEL_NAME = "roberta-base"
    MAX_LENGTH = 256
    BATCH_SIZE = 64
    NUM_TRAIN_EPOCHS = 30
    LEARNING_RATE = 2e-5
    MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL = 2

    # === OOD 균형화 설정 ===
    OOD_SAMPLING_STRATEGY = 'original'
    OOD_SAMPLE_SIZE = 500
    OOD_PROPORTIONAL_RATIO = 0.1
    REPORT_OOD_BALANCE_STATS = True
    
    # === 하드웨어 설정 ===
    ACCELERATOR = "auto"
    DEVICES = "auto"
    PRECISION = "16-mixed" if torch.cuda.is_available() else "32-true"
    NUM_WORKERS = min(2, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1)

    # === 학습 제어 ===
    LOG_EVERY_N_STEPS = 50
    GRADIENT_CLIP_VAL = 1.0
    USE_WEIGHTED_LOSS = True
    USE_LR_SCHEDULER = True
    RANDOM_STATE = 42

    # === 어텐션 설정 ===
    ATTENTION_TOP_PERCENT = 0.05
    MIN_TOP_WORDS = 1
    TOP_K_ATTENTION = 3
    #ATTENTION_LAYER = [-1]
    ATTENTION_LAYER = [-1, -3, -6]

    # === OE 필터링 설정 ===
    USE_ELBOW_METHOD = True
    METRIC_SETTINGS = {
        'attention_entropy': {'percentile': 75, 'mode': 'higher'},
        'max_attention': {'percentile': 15, 'mode': 'lower'},
        'removed_avg_attention': {'percentile': 85, 'mode': 'higher'},
        'top_k_avg_attention': {'percentile': 25, 'mode': 'lower'}
    }
    FILTERING_SEQUENCE = [
            ('attention_entropy', {'percentile': 75, 'mode': 'higher'}),
        ('removed_avg_attention', {'percentile': 85, 'mode': 'higher'}),
        ('max_attention', {'percentile': 15, 'mode': 'lower'})
    ]
    TEXT_COLUMN_IN_OE_FILES = 'masked_text_attention'

    # === OSR Experiment Settings ===
    OSR_EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "osr_experiments")
    OSR_MODEL_DIR = os.path.join(OSR_EXPERIMENT_DIR, "models")
    OSR_RESULT_DIR = os.path.join(OSR_EXPERIMENT_DIR, "results")

    OSR_MODEL_TYPE = "roberta-base"
    OSR_MAX_LENGTH = MAX_LENGTH
    OSR_BATCH_SIZE = 16
    OSR_NUM_EPOCHS = 20
    OSR_LEARNING_RATE = LEARNING_RATE

    # === 다중 외부 OE 데이터셋 설정 ===
    OSR_EXTERNAL_OE_DATASETS = ['wikitext', '20newsgroups', 'sst2', 'trec', 'wmt16']
    NLP_OOD_EVAL_DATASETS = ['wmt16', 'wikitext', 'trec', 'sst2', 'syslog']

    # Common OSR settings
    OSR_OE_LAMBDA = 0.5
    OSR_TEMPERATURE = 2.0
    OSR_THRESHOLD_PERCENTILE = 5.0
    OSR_NUM_DATALOADER_WORKERS = min(2, NUM_WORKERS)

    # Early stopping 설정
    BASE_MODEL_EARLY_STOPPING_PATIENCE = 5
    OSR_EARLY_STOPPING_PATIENCE = 5
    OSR_EARLY_STOPPING_MIN_DELTA = 0.001

    # === 양 통제 실험 설정 ===
    OSR_FAIR_COMPARISON_MODE = False

    # === 실행 단계 제어 ===
    STAGE_MODEL_TRAINING = True
    STAGE_ATTENTION_EXTRACTION = True
    STAGE_OE_EXTRACTION = True
    STAGE_VISUALIZATION = True
    STAGE_OSR_EXPERIMENTS = True

    # === Flags ===
    OSR_EVAL_ONLY = False
    OSR_FORCE_RETRAIN = False
    OSR_NO_PLOT_PER_EXPERIMENT = False
    OSR_SKIP_STANDARD_MODEL = False

    # === HuggingFace Cache 분리 ===
    HUGGINGFACE_CACHE_DIR = "huggingface_cache"

    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        dirs = [
            cls.OUTPUT_DIR, cls.MODEL_SAVE_DIR, cls.LOG_DIR,
            cls.CONFUSION_MATRIX_DIR, cls.VIS_DIR, cls.OE_DATA_DIR,
            cls.ATTENTION_DATA_DIR,
            cls.OSR_EXPERIMENT_DIR, cls.OSR_MODEL_DIR, cls.OSR_RESULT_DIR,
            cls.HUGGINGFACE_CACHE_DIR
        ]
        for dir_path in dirs:
            ensure_directory(dir_path)

    @classmethod
    def update_paths_for_dataset(cls, dataset_name: str, base_dir: str = None):
        """특정 데이터셋에 맞게 경로 업데이트"""
        if base_dir is None:
            base_dir = cls.BASE_OUTPUT_DIR
            
        dataset_output_dir = os.path.join(base_dir, f"dataset_{dataset_name}")
        cls.OUTPUT_DIR = dataset_output_dir
        cls.MODEL_SAVE_DIR = os.path.join(dataset_output_dir, "base_classifier_model")
        cls.LOG_DIR = os.path.join(dataset_output_dir, "lightning_logs")
        cls.CONFUSION_MATRIX_DIR = os.path.join(cls.LOG_DIR, "confusion_matrices")
        cls.VIS_DIR = os.path.join(dataset_output_dir, "oe_extraction_visualizations")
        cls.OE_DATA_DIR = os.path.join(dataset_output_dir, "extracted_oe_datasets")
        cls.ATTENTION_DATA_DIR = os.path.join(dataset_output_dir, "attention_analysis")
        cls.OSR_EXPERIMENT_DIR = os.path.join(dataset_output_dir, "osr_experiments")
        cls.OSR_MODEL_DIR = os.path.join(cls.OSR_EXPERIMENT_DIR, "models")
        cls.OSR_RESULT_DIR = os.path.join(cls.OSR_EXPERIMENT_DIR, "results")

    @classmethod
    def save_config(cls, filepath=None):
        """설정을 JSON 파일로 저장"""
        if filepath is None:
            filepath = os.path.join(cls.OUTPUT_DIR, 'config_enhanced.json')

        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    config_dict[attr] = value
                elif isinstance(value, tuple):
                    config_dict[attr] = list(value)

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {filepath}")

# === 헬퍼 함수들 ===
def set_seed(seed: int):
    """시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    print(f"Seed set to {seed}")

def clear_memory():
    """GPU와 일반 메모리를 정리하는 헬퍼 함수"""
    print("\nAttempting to clear memory...")
    try:
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    except Exception as e:
        print(f"Warning: Error during memory cleanup: {e}")
    
    print("🧹 Memory cleared.")

def preprocess_text_for_nlp(text):
    """NLP를 위한 텍스트 전처리"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_text_for_roberta(text):
    """RoBERTa를 위한 텍스트 전처리"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_nltk(text):
    """NLTK를 사용한 토큰화"""
    if not text: 
        return []
    
    global _NLTK_DOWNLOADS_DONE
    if not _NLTK_DOWNLOADS_DONE: 
        ensure_nltk_data()
    
    try:
        return word_tokenize(text)
    except LookupError as e:
        print(f"NLTK tokenization failed: {e}. Using simple split.")
        return text.split()
    except Exception as e:
        print(f"NLTK tokenization failed: {e}. Using simple split.")
        return text.split()
    
def create_masked_sentence(original_text, important_words):
    """중요 단어를 제거하여 마스킹된 문장 생성"""
    if not isinstance(original_text, str): return ""
    if not important_words: return original_text

    processed_text = preprocess_text_for_nlp(original_text)
    tokens = tokenize_nltk(processed_text)
    important_set_lower = {word.lower() for word in important_words}

    masked_tokens = [word for word in tokens if word.lower() not in important_set_lower]
    masked_sentence = ' '.join(masked_tokens)

    return masked_sentence if masked_sentence else "__EMPTY_MASKED__"

def safe_literal_eval(val):
    """문자열을 리스트로 안전하게 변환"""
    try:
        if isinstance(val, str) and val.strip().startswith('['):
            return ast.literal_eval(val)
        elif isinstance(val, list):
            return val
    except (ValueError, SyntaxError):
        pass
    return []

def find_elbow_point(scores: np.ndarray) -> int:
    """올바른 Elbow Point Detection 구현"""
    if len(scores) < 3:
        return len(scores) // 2

    y = np.sort(scores)
    n_points = len(y)
    x = np.arange(n_points)

    # 시작점과 끝점
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    # 직선 벡터
    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)
    
    if line_length == 0:
        return len(scores) // 2

    # 각 점에서 직선까지의 거리 계산
    distances = []
    for i in range(n_points):
        point = np.array([x[i], y[i]])
        point_vec = point - p1
        
        # 직선까지의 수직 거리 계산
        if line_length > 0:
            cross = np.abs(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0])
            distance = cross / line_length
        else:
            distance = 0
        distances.append(distance)

    elbow_index = np.argmax(distances)
    return int(elbow_index)

# === NLP 데이터셋 로더들 ===
class NLPDatasetLoader:
    @staticmethod
    def _load_and_extract(name, subset=None, text_col='text', label_col='label', split='train'):
        print(f"Loading dataset: {name} (subset: {subset}, split: {split})")
        try:
            # 데이터셋 로드
            dataset = load_dataset(name, subset, cache_dir=Config.HUGGINGFACE_CACHE_DIR)
            
            # SST2의 경우 test split에 라벨이 없으므로 validation 사용
            if name == 'glue' and subset == 'sst2' and split == 'test':
                print(f"  - SST2: Using 'validation' split instead of 'test' (test has no labels)")
                split = 'validation'
                
            if split not in dataset:
                available_splits = list(dataset.keys())
                print(f"  - Available splits: {available_splits}")
                
                if 'validation' in available_splits and split == 'test':
                    split = 'validation'
                elif available_splits:
                    split = available_splits[0]
                else:
                    raise ValueError(f"No splits found for {name}/{subset}")
                print(f"  - Split '{split}' not found, using '{split}' instead.")

            # Handle translation datasets (like WMT)
            if 'translation' in dataset[split].column_names:
                text_col = Config.NLP_DATASETS['wmt16']['text_column']
                texts = [item[text_col] for item in dataset[split]['translation'] if text_col in item and item[text_col]]
                labels = None
            else:
                # Extract texts
                texts = []
                labels = []
                
                for item in dataset[split]:
                    text_value = item.get(text_col)
                    if text_value:
                        texts.append(str(text_value))
                        
                        # Extract label if available
                        if label_col and label_col in item:
                            label_value = item.get(label_col)
                            if label_value is not None:
                                labels.append(label_value)
                            else:
                                labels.append(-1)
                        else:
                            labels = None
                            break

            # Validate labels
            if labels is not None and all(l == -1 for l in labels):
                print(f"  - Warning: All labels are missing for {name}/{split}")
                labels = None

            print(f"  - Loaded {len(texts)} samples" + (f" with {len([l for l in labels if l != -1])} labeled" if labels else ""))
            return {'text': texts, 'label': labels}
            
        except Exception as e:
            print(f"Error loading {name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def load_any_dataset(dataset_key: str, split: str = 'train'):
        if dataset_key not in Config.NLP_DATASETS:
            raise ValueError(f"Dataset key '{dataset_key}' not found in Config.NLP_DATASETS")

        params = Config.NLP_DATASETS[dataset_key]

        # Special handling for specific datasets
        if dataset_key == 'wikitext':
            return NLPDatasetLoader.load_wikitext(split)
        else:
            return NLPDatasetLoader._load_and_extract(
                name=params['name'],
                subset=params.get('subset'),
                text_col=params['text_column'],
                label_col=params.get('label_column'),
                split=split
            )

    @staticmethod
    def load_wikitext(split='train'):
        print(f"Loading WikiText-2 dataset (split: {split})...")
        try:
            dataset = load_dataset("wikitext", 'wikitext-2-raw-v1', cache_dir=Config.HUGGINGFACE_CACHE_DIR)
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in WikiText-2.")

            texts = []
            for item in dataset[split]:
                text_content = item['text'].strip()
                if text_content:
                    sentences = sent_tokenize(text_content)
                    for sent in sentences:
                        sent_clean = sent.strip()
                        if len(sent_clean) > 10 and not (sent_clean.startswith(" =") and sent_clean.endswith("= ")):
                            texts.append(sent_clean)

            print(f"Loaded {len(texts)} sentences from WikiText-2 '{split}' split.")
            return {'text': texts, 'label': None}
        except Exception as e:
            print(f"Error loading WikiText-2: {e}")
            return None

# === NLP용 Dataset 클래스 ===
class NLPDataset(TorchDataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else 0

        encoding = self.tokenizer(
            preprocess_text_for_roberta(text),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.long(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# === OSR을 위한 NLP Dataset ===
class OSRNLPDataset(TorchDataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            preprocess_text_for_roberta(text),
            truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'text': text,
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.long(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# === OSR을 위한 Syslog Dataset ===
class OSRSyslogTextDataset(TorchDataset):
    """PyTorch Dataset for Syslog text data for OSR tasks."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            preprocess_text_for_roberta(text),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# === OOD 균형화 함수들 ===
def balance_ood_dataset(dataset, target_size: int, dataset_name: str, random_state: int = 42):
    """단일 OOD 데이터셋을 목표 크기로 균형화 - 개선된 버전"""
    current_size = len(dataset)
    
    if current_size == target_size:
        return dataset
        
    np.random.seed(random_state)
    
    # 타입별로 적절한 처리
    if hasattr(dataset, 'texts') and hasattr(dataset, 'labels'):
        # Custom dataset 타입
        if current_size > target_size:
            # 다운샘플링
            indices = np.random.choice(current_size, target_size, replace=False)
        else:
            # 업샘플링
            indices = np.random.choice(current_size, target_size, replace=True)
            
        sampled_texts = [dataset.texts[i] for i in indices]
        sampled_labels = [dataset.labels[i] for i in indices]
        
        # 동일한 타입으로 재생성
        return type(dataset)(
            sampled_texts, 
            sampled_labels, 
            dataset.tokenizer if hasattr(dataset, 'tokenizer') else None,
            dataset.max_length if hasattr(dataset, 'max_length') else getattr(dataset, 'max_len', 256)
        )
    else:
        # Subset 또는 기타 타입
        if current_size > target_size:
            indices = np.random.choice(current_size, target_size, replace=False)
        else:
            indices = np.random.choice(current_size, target_size, replace=True)
        return Subset(dataset, indices)

def compute_balance_target_sizes(ood_datasets: Dict[str, Dict], strategy: str, config: Config, id_size: int) -> Dict[str, int]:
    """균형화 전략에 따른 각 OOD 데이터셋의 목표 크기 계산"""
    original_sizes = {name: data['count'] for name, data in ood_datasets.items()}
    
    if strategy == 'original':
        return original_sizes
    elif strategy == 'min':
        min_size = min(original_sizes.values())
        return {name: min_size for name in original_sizes}
    elif strategy == 'fixed':
        return {name: config.OOD_SAMPLE_SIZE for name in original_sizes}
    elif strategy == 'proportional':
        target_size = int(id_size * config.OOD_PROPORTIONAL_RATIO)
        return {name: target_size for name in original_sizes}
    else:
        raise ValueError(f"Unknown OOD sampling strategy: {strategy}")

def print_ood_balance_stats(original_sizes: Dict[str, int], target_sizes: Dict[str, int], balanced_sizes: Dict[str, int]):
    """OOD 균형화 전후 통계 출력"""
    print("\n=== OOD Dataset Balancing Statistics ===")
    print(f"{'Dataset':<20} {'Original':<10} {'Target':<10} {'Balanced':<10} {'Change':<10}")
    print("-" * 60)
    
    for name in original_sizes:
        orig = original_sizes[name]
        target = target_sizes[name]
        balanced = balanced_sizes.get(name, orig)
        change = f"{((balanced/orig - 1) * 100):+.1f}%" if orig > 0 else "N/A"
        print(f"{name:<20} {orig:<10} {target:<10} {balanced:<10} {change:<10}")
    
    print(f"\nTotal OOD samples: {sum(original_sizes.values())} → {sum(balanced_sizes.values())}")
    print(f"Min size: {min(original_sizes.values())} → {min(balanced_sizes.values())}")
    print(f"Max size: {max(original_sizes.values())} → {max(balanced_sizes.values())}")
    print(f"Std dev: {np.std(list(original_sizes.values())):.1f} → {np.std(list(balanced_sizes.values())):.1f}")

# === Enhanced DataModule ===
class EnhancedDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config'])

        self.tokenizer = None
        self.data_collator = None

        self.df_full = None
        self.train_df_final = None
        self.val_df_final = None
        self.label2id = None
        self.id2label = None
        self.num_labels = None
        self.class_weights = None

        self.tokenized_train_val_datasets = None

    def prepare_data(self):
        """Data download etc."""
        if self.config.EXPERIMENT_MODE == "nlp":
            print("Preparing NLP data (downloading if necessary)...")
            try:
                for ds_name in self.config.NLP_DATASETS.keys():
                    params = self.config.NLP_DATASETS[ds_name]
                    load_dataset(
                        params['name'],
                        params.get('subset'),
                        cache_dir=self.config.HUGGINGFACE_CACHE_DIR
                    )
                print("NLP data preparation checks complete.")
            except Exception as e:
                print(f"Warning: Error during NLP data preparation: {e}")
        else:
            print("Syslog mode: Data preparation step skipped (using local files).")

    def setup(self, stage: Optional[str] = None):
        """Load data, preprocess, split, tokenize."""
        if self.df_full is not None and self.tokenizer is not None:
             print("DataModule already set up.")
             return

        if self.config.EXPERIMENT_MODE == "nlp":
            self._setup_nlp()
        else:
            self._setup_syslog()

    def _setup_nlp(self):
        print(f"Setting up DataModule for NLP mode: {self.config.CURRENT_NLP_DATASET}")
        dataset_name = self.config.CURRENT_NLP_DATASET

        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Load train data
        train_data = NLPDatasetLoader.load_any_dataset(dataset_name, split='train')
        
        # Try to load test/validation data
        test_data = None
        validation_data = None
        
        try:
            test_data = NLPDatasetLoader.load_any_dataset(dataset_name, split='test')
        except:
            print(f"No test split found for {dataset_name}")
        
        try:
            validation_data = NLPDatasetLoader.load_any_dataset(dataset_name, split='validation')
        except:
            print(f"No validation split found for {dataset_name}")

        if train_data is None or train_data['text'] is None:
            raise ValueError(f"Failed to load train data for ID dataset: {dataset_name}")

        train_df = pd.DataFrame(train_data)
        
        # Handle validation/test data
        val_df = None
        
        if validation_data and validation_data.get('label') is not None:
            val_df = pd.DataFrame(validation_data)
            print(f"Using validation split for evaluation")
        elif test_data and test_data.get('label') is not None:
            val_df = pd.DataFrame(test_data)
            print(f"Using test split for evaluation")
        
        if val_df is None or len(val_df) == 0:
            print(f"No labeled validation/test data found. Splitting train data 80/20.")
            train_df, val_df = train_test_split(
                train_df, 
                test_size=0.2, 
                random_state=self.config.RANDOM_STATE,
                stratify=train_df['label'] if 'label' in train_df.columns else None
            )
        
        # Set up splits
        train_df['split'] = 'train'
        val_df['split'] = 'id_test'
        self.df_full = pd.concat([train_df, val_df], ignore_index=True)

        # Create label mappings
        unique_labels_train = sorted([int(label) for label in train_df['label'].dropna().unique()])
        self.label2id = {int(label): int(i) for i, label in enumerate(unique_labels_train)}
        self.id2label = {int(i): int(label) for label, i in self.label2id.items()}
        self.num_labels = len(unique_labels_train)

        print(f"NLP Label mapping: {self.label2id}")
        
        # Map labels to IDs
        self.df_full['label_id'] = self.df_full['label'].map(self.label2id).fillna(-1).astype(int)

        # Final train/val dataframes
        self.train_df_final = self.df_full[self.df_full['split'] == 'train'].copy()
        self.val_df_final = self.df_full[self.df_full['split'] == 'id_test'].copy()
        self.val_df_final = self.val_df_final[self.val_df_final['label_id'] != -1]

        # Compute class weights
        if self.config.USE_WEIGHTED_LOSS:
            self._compute_class_weights_nlp(self.train_df_final)

        print(f"NLP DataModule setup: Train samples: {len(self.train_df_final)}, Val/ID-Test samples: {len(self.val_df_final)}")
        
        if len(self.val_df_final) == 0:
            raise ValueError(f"No validation data available for {dataset_name}. Cannot proceed with training.")
        
        self._tokenize_datasets_nlp()

    def _setup_syslog(self):
        print(f"Setting up DataModule for Syslog mode: {self.config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        print(f"Loading Syslog data from {self.config.ORIGINAL_DATA_PATH}")
        self.df_full = safe_load_dataset_file(self.config.ORIGINAL_DATA_PATH)

        required_cols = [self.config.TEXT_COLUMN, self.config.CLASS_COLUMN]
        if not all(col in self.df_full.columns for col in required_cols):
            raise ValueError(f"Missing columns in {self.config.ORIGINAL_DATA_PATH}: {required_cols}")

        self.df_full = self.df_full.dropna(subset=[self.config.TEXT_COLUMN, self.config.CLASS_COLUMN])
        self.df_full[self.config.CLASS_COLUMN] = self.df_full[self.config.CLASS_COLUMN].astype(str).str.lower()
        self.df_full[self.config.TEXT_COLUMN] = self.df_full[self.config.TEXT_COLUMN].astype(str)

        exclude_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
        df_known = self.df_full[self.df_full[self.config.CLASS_COLUMN] != exclude_class_lower].copy()

        known_classes_str = sorted(df_known[self.config.CLASS_COLUMN].unique())
        self.label2id = {str(label): int(i) for i, label in enumerate(known_classes_str)}
        self.id2label = {int(i): str(label) for label, i in self.label2id.items()}
        self.num_labels = len(known_classes_str)

        if self.num_labels == 0: raise ValueError("No known classes for Syslog.")
        print(f"Syslog Label mapping: {self.label2id}")

        df_known['label'] = df_known[self.config.CLASS_COLUMN].map(self.label2id)

        label_counts = df_known['label'].value_counts()
        valid_labels = label_counts[label_counts >= self.config.MIN_SAMPLES_PER_CLASS_FOR_TRAIN_VAL].index
        self.df_known_for_train_val = df_known[df_known['label'].isin(valid_labels)].copy()

        if len(valid_labels) < self.num_labels:
            final_classes_str = sorted(self.df_known_for_train_val[self.config.CLASS_COLUMN].unique())
            self.label2id = {str(label): int(i) for i, label in enumerate(final_classes_str)}
            self.id2label = {int(i): str(label) for label, i in self.label2id.items()}
            self.num_labels = len(final_classes_str)
            self.df_known_for_train_val['label'] = self.df_known_for_train_val[self.config.CLASS_COLUMN].map(self.label2id)
            print(f"  Syslog Updated label mapping: {self.num_labels} classes. {self.label2id}")

        if len(self.df_known_for_train_val) == 0:
            raise ValueError("No Syslog data available after filtering.")

        if self.config.USE_WEIGHTED_LOSS:
            self._compute_class_weights_syslog(self.df_known_for_train_val)

        self._split_train_val_syslog()
        self._tokenize_datasets_syslog()

    def _compute_class_weights_nlp(self, train_df):
        labels_for_weights = train_df['label_id'].values
        unique_labels_present = np.unique(labels_for_weights)

        if len(unique_labels_present) < self.num_labels:
             print(f"Warning: Not all {self.num_labels} classes are present.")
        if len(unique_labels_present) <=1:
            print("Warning: Only one or no class present. Using uniform weights.")
            self.class_weights = None
            return

        try:
            class_weights_array = compute_class_weight('balanced', classes=unique_labels_present, y=labels_for_weights)
            self.class_weights = torch.ones(self.num_labels)
            for i, label_idx in enumerate(unique_labels_present):
                if 0 <= label_idx < self.num_labels:
                    self.class_weights[label_idx] = class_weights_array[i]
            print(f"Computed NLP class weights: {self.class_weights}")
        except ValueError as e:
            print(f"Error computing NLP class weights: {e}. Using uniform weights.")
            self.class_weights = None

    def _compute_class_weights_syslog(self, df_known_for_train_val):
        labels_for_weights = df_known_for_train_val['label'].values
        unique_labels_present = np.unique(labels_for_weights)

        if len(unique_labels_present) < self.num_labels:
             print(f"Warning: Not all {self.num_labels} classes are present.")
        if len(unique_labels_present) <=1:
            print("Warning: Only one or no class present. Using uniform weights.")
            self.class_weights = None
            return

        try:
            class_weights_array = compute_class_weight('balanced', classes=unique_labels_present, y=labels_for_weights)
            self.class_weights = torch.ones(self.num_labels)
            for i, class_idx in enumerate(unique_labels_present):
                 if 0 <= class_idx < self.num_labels:
                    self.class_weights[class_idx] = class_weights_array[i]
            print(f"Computed Syslog class weights: {self.class_weights}")
        except ValueError as e:
            print(f"Error computing Syslog class weights: {e}. Using uniform weights.")
            self.class_weights = None

    def _split_train_val_syslog(self):
        min_class_count = self.df_known_for_train_val['label'].value_counts().min()
        stratify_col = self.df_known_for_train_val['label'] if min_class_count > 1 else None

        self.train_df_final, self.val_df_final = train_test_split(
            self.df_known_for_train_val, test_size=0.2,
            random_state=self.config.RANDOM_STATE, stratify=stratify_col
        )
        print(f"Syslog split - Train: {len(self.train_df_final)}, Val/ID-Test: {len(self.val_df_final)}")

    def _tokenize_datasets_syslog(self):
        raw_datasets = DatasetDict({
            'train': Dataset.from_pandas(self.train_df_final),
            'validation': Dataset.from_pandas(self.val_df_final)
        })

        def tokenize_fn(examples):
            return self.tokenizer(
                [preprocess_text_for_roberta(text) for text in examples[self.config.TEXT_COLUMN]],
                truncation=True, padding=False, max_length=self.config.MAX_LENGTH
            )

        self.tokenized_train_val_datasets = raw_datasets.map(
            tokenize_fn, batched=True,
            num_proc=max(1, self.config.NUM_WORKERS // 2),
            remove_columns=[col for col in raw_datasets['train'].column_names if col not in ['label', 'input_ids', 'attention_mask']]
        )
        self.tokenized_train_val_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    def _tokenize_datasets_nlp(self):
        """NLP 모드에서 토큰화 처리"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized for NLP mode")
        print("NLP tokenization will be handled by NLPDataset class")

    def train_dataloader(self):
        if self.config.EXPERIMENT_MODE == "nlp":
            dataset = NLPDataset(
                self.train_df_final['text'].tolist(),
                self.train_df_final['label_id'].tolist(),
                self.tokenizer,
                max_length=self.config.MAX_LENGTH
            )
            return create_safe_dataloader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS
            )
        else:
            return create_safe_dataloader(
                self.tokenized_train_val_datasets['train'],
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                collate_fn=self.data_collator
            )

    def val_dataloader(self):
        if self.config.EXPERIMENT_MODE == "nlp":
            dataset = NLPDataset(
                self.val_df_final['text'].tolist(),
                self.val_df_final['label_id'].tolist(),
                self.tokenizer,
                max_length=self.config.MAX_LENGTH
            )
            return create_safe_dataloader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.NUM_WORKERS
            )
        else:
            return create_safe_dataloader(
                self.tokenized_train_val_datasets['validation'],
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.NUM_WORKERS,
                collate_fn=self.data_collator
            )

    def get_full_dataframe(self):
        if self.df_full is None: self.setup()
        return self.df_full

# === Enhanced Model (LightningModule) ===
class EnhancedModel(pl.LightningModule):
    def __init__(self, config: Config, num_labels: int, label2id: dict, id2label: dict,
                class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.config_params = config

        self.label2id = convert_numpy_types(label2id)
        self.id2label = convert_numpy_types(id2label)
        self.num_labels = int(num_labels)

        try:
            self.save_hyperparameters(
                'num_labels',
                'label2id',
                'id2label',
                ignore=['config', 'config_params', 'class_weights']
            )
        except Exception as e:
            print(f"Warning: hyperparameter save failed: {e}")
            self.save_hyperparameters(ignore=['config', 'config_params', 'class_weights'])

        self.class_weights = class_weights

        model_cache_dir = self.config_params.HUGGINGFACE_CACHE_DIR

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
            output_attentions=True,
            output_hidden_states=True,
            cache_dir=model_cache_dir
        )
        print(f"Initialized Transformer classifier ({config.MODEL_NAME}) for {num_labels} classes.")

        if self.config_params.USE_WEIGHTED_LOSS and self.class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            print(f"{config.EXPERIMENT_MODE} classifier using weighted CrossEntropyLoss.")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print(f"{config.EXPERIMENT_MODE} classifier using standard CrossEntropyLoss.")

        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels, average='micro'),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='weighted'),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_labels)

    def setup(self, stage=None):
        if self.config_params.USE_WEIGHTED_LOSS and hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)
            print(f"Moved class weights for loss_fn to {self.device}")

    def forward(self, batch, output_features=False, output_attentions=False):
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        if input_ids is None: raise ValueError("Batch missing 'input_ids'")

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_features,
            output_attentions=output_attentions
        )

    def _common_step(self, batch, batch_idx):
        if 'label' in batch:
            labels = batch.pop('label')
        elif 'labels' in batch:
            labels = batch.pop('labels')
        else:
            raise KeyError("No 'label' or 'labels' found in batch")

        batch['labels'] = labels
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.train_metrics.update(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.val_metrics.update(preds, labels)
        self.val_cm.update(preds, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def on_train_epoch_end(self):
        try:
            computed_metrics = self.train_metrics.compute()
            self.log_dict(computed_metrics, prog_bar=True)
            self.train_metrics.reset()
        except Exception as e:
            print(f"Warning: Train metrics error: {e}")
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        val_metrics_output = self.val_metrics.compute()
        self.log_dict(val_metrics_output, prog_bar=True)
        self.val_metrics.reset()

        try:
            val_cm_computed = self.val_cm.compute()
            class_names = [str(self.id2label.get(i, f"ID_{i}")) for i in range(self.num_labels)]

            if val_cm_computed.shape[0] != len(class_names):
                print(f"Warning: CM dim ({val_cm_computed.shape[0]}) != class_names len ({len(class_names)}). Adjusting.")
                class_names = [f"Class_{i}" for i in range(val_cm_computed.shape[0])]

            cm_df = pd.DataFrame(val_cm_computed.cpu().numpy(), index=class_names, columns=class_names)
            print(f"\nClassifier Validation Confusion Matrix (Epoch {self.current_epoch}):")
            print(cm_df)
            cm_filename = os.path.join(self.config_params.CONFUSION_MATRIX_DIR,
                                     f"clf_val_cm_epoch_{self.current_epoch}.csv")
            cm_df.to_csv(cm_filename)
        except Exception as e:
            print(f"Error in classifier validation CM: {e}")
        finally:
            self.val_cm.reset()

    def configure_optimizers(self):
        lr = self.config_params.LEARNING_RATE
        optimizer = AdamW(self.parameters(), lr=lr, eps=1e-8)

        if not self.config_params.USE_LR_SCHEDULER:
            return optimizer

        num_training_steps = self.trainer.estimated_stepping_batches
        if num_training_steps is None or num_training_steps <= 0:
             num_training_steps = (len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches) * self.trainer.max_epochs

        if num_training_steps <= 0:
            print("Warning: Could not determine num_training_steps for LR scheduler. Using fixed steps.")
            num_training_steps = 10000

        num_warmup_steps = int(num_training_steps * 0.1)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }

# === Enhanced Attention Analyzer ===
class EnhancedAttentionAnalyzer:
    def __init__(self, config: Config, model_pl: EnhancedModel, tokenizer, device):
        self.config = config
        self.model_pl = model_pl.to(device)
        self.model_pl.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.is_nlp_mode = (config.EXPERIMENT_MODE == "nlp")
        self.attention_layers = config.ATTENTION_LAYER if isinstance(config.ATTENTION_LAYER, list) else [config.ATTENTION_LAYER]
        print(f"Attention Analyzer will use layers: {self.attention_layers}")

    @torch.no_grad()
    def get_word_attention_scores(self, texts: List[str]) -> List[Dict[str, float]]:
        self.model_pl.eval()
        self.model_pl.to(self.device)

        batch_size = self.config.BATCH_SIZE
        all_word_scores = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Computing word attention scores", leave=False):
            batch_texts = texts[i:i+batch_size]
            batch_scores = self._process_attention_batch_transformer(batch_texts)
            all_word_scores.extend(batch_scores)
        return all_word_scores

    def _process_attention_batch_transformer(self, batch_texts: List[str]) -> List[Dict[str, float]]:
        if not batch_texts: return []

        processed_texts = [preprocess_text_for_roberta(text) for text in batch_texts]

        inputs = self.tokenizer(
            processed_texts, return_tensors='pt', truncation=True,
            max_length=self.config.MAX_LENGTH,
            padding=True, return_offsets_mapping=True
        )
        offset_mappings = inputs.pop('offset_mapping').cpu().numpy()
        input_ids_batch = inputs['input_ids'].cpu().numpy()

        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model_pl.forward(inputs_on_device, output_attentions=True)

        # 여러 레이어의 어텐션을 평균
        avg_attentions_batch = torch.stack([outputs.attentions[l] for l in self.attention_layers]).mean(dim=0)
        avg_attentions_batch_np = avg_attentions_batch.cpu().numpy()

        batch_word_scores = []
        for i in range(len(batch_texts)):
            word_scores = self._extract_word_scores_from_transformer_attention(
                avg_attentions_batch_np[i],
                input_ids_batch[i],
                offset_mappings[i],
                processed_texts[i]
            )
            batch_word_scores.append(word_scores)

        del inputs, inputs_on_device, outputs, avg_attentions_batch, avg_attentions_batch_np
        clear_memory()
        return batch_word_scores

    def _extract_word_scores_from_transformer_attention(self, attention_sample_layer, input_ids_single, offset_mapping_single, original_text_single):
        cls_attentions_to_tokens = np.mean(attention_sample_layer[:, 0, :], axis=0)

        word_scores = defaultdict(list)
        current_word_tokens_indices = []
        last_word_end_offset = 0

        # 길이 검증
        min_length = min(len(input_ids_single), len(offset_mapping_single))
        if min_length == 0:
            return {}

        for token_idx in range(min_length):
            token_id_val = input_ids_single[token_idx]
            offset = offset_mapping_single[token_idx]
            
            # offset 검증
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                continue
                
            # special token이거나 empty offset인 경우
            if offset[0] == offset[1] or token_id_val in self.tokenizer.all_special_ids:
                if current_word_tokens_indices:
                    self._finalize_current_word(
                        current_word_tokens_indices, offset_mapping_single, 
                        original_text_single, cls_attentions_to_tokens, word_scores
                    )
                    current_word_tokens_indices = []
                last_word_end_offset = offset[1]
                continue

            # 새로운 단어 시작 검사
            is_new_word_start = (offset[0] > last_word_end_offset) or not current_word_tokens_indices

            if is_new_word_start and current_word_tokens_indices:
                self._finalize_current_word(
                    current_word_tokens_indices, offset_mapping_single, 
                    original_text_single, cls_attentions_to_tokens, word_scores
                )
                current_word_tokens_indices = []

            current_word_tokens_indices.append(token_idx)
            last_word_end_offset = offset[1]

        # 마지막 단어 처리
        if current_word_tokens_indices:
            self._finalize_current_word(
                current_word_tokens_indices, offset_mapping_single, 
                original_text_single, cls_attentions_to_tokens, word_scores
            )

        return {word: np.mean(scores) for word, scores in word_scores.items()}

    def _finalize_current_word(self, token_indices, offset_mapping, original_text, attentions, word_scores):
        """현재 단어의 attention score를 계산하고 word_scores에 추가"""
        if not token_indices:
            return
            
        try:
            start_char = offset_mapping[token_indices[0]][0]
            end_char = offset_mapping[token_indices[-1]][1]
            
            # 텍스트 범위 검증
            if start_char >= len(original_text) or end_char > len(original_text) or start_char >= end_char:
                return
                
            word_str = original_text[start_char:end_char]
            
            # attention score 계산 시 인덱스 범위 검증
            valid_token_indices = [idx for idx in token_indices if idx < len(attentions)]
            if valid_token_indices:
                avg_score_for_word = np.mean(attentions[valid_token_indices])
                if word_str.strip():
                    word_scores[word_str.strip()].append(avg_score_for_word)
        except (IndexError, TypeError) as e:
            print(f"Warning: Error processing word tokens: {e}")

    def extract_top_attention_words(self, word_scores_dict: Dict[str, float]) -> List[str]:
        if not word_scores_dict: return []

        sorted_words = sorted(word_scores_dict.items(), key=lambda x: x[1], reverse=True)
        num_total_words = len(sorted_words)
        n_top = max(self.config.MIN_TOP_WORDS, math.ceil(num_total_words * self.config.ATTENTION_TOP_PERCENT))

        top_n_word_score_pairs = sorted_words[:n_top]

        try:
            stop_words_set = set(stopwords.words('english'))
        except LookupError:
            print("NLTK stopwords not found, using a basic list.")
            stop_words_set = {'a', 'an', 'the', 'is', 'was', 'to', 'of', 'for', 'on', 'in', 'at', 'and', 'or', 'it', 's'}

        top_words_filtered = [word for word, score in top_n_word_score_pairs
                             if word.lower() not in stop_words_set and len(word) > 1]

        if not top_words_filtered and top_n_word_score_pairs:
            return [word for word, score in top_n_word_score_pairs]
        return top_words_filtered

    def process_full_dataset(self, df: pd.DataFrame, exclude_class: Optional[str] = None) -> pd.DataFrame:
        print("Processing full dataset for attention analysis...")
        df_to_process = df.copy()

        if exclude_class:
            col_to_check = 'label' if self.is_nlp_mode else self.config.CLASS_COLUMN
            exclude_class_lower = exclude_class.lower()
            if col_to_check in df_to_process.columns:
                 df_to_process[col_to_check] = df_to_process[col_to_check].astype(str)
                 analysis_mask = df_to_process[col_to_check].str.lower() != exclude_class_lower
            else:
                 print(f"Warning: Column '{col_to_check}' not found for excluding class. Analyzing all rows.")
                 analysis_mask = pd.Series([True] * len(df_to_process), index=df_to_process.index)
        else:
            analysis_mask = pd.Series([True] * len(df_to_process), index=df_to_process.index)

        texts_for_analysis = df_to_process.loc[analysis_mask, self.config.TEXT_COLUMN].tolist()
        indices_analyzed = df_to_process.index[analysis_mask]

        if not texts_for_analysis:
            print("No texts to analyze after filtering.")
            df_to_process['top_attention_words'] = [[] for _ in range(len(df_to_process))]
            df_to_process[self.config.TEXT_COLUMN_IN_OE_FILES] = df_to_process[self.config.TEXT_COLUMN]
            return df_to_process

        print(f"Computing word attention scores for {len(texts_for_analysis)} samples...")
        all_word_scores_list = self.get_word_attention_scores(texts_for_analysis)

        df_to_process['top_attention_words'] = pd.Series([[] for _ in range(len(df_to_process))], index=df_to_process.index, dtype=object)
        df_to_process[self.config.TEXT_COLUMN_IN_OE_FILES] = df_to_process[self.config.TEXT_COLUMN]

        print("Extracting top attention words and creating masked texts...")
        for i, original_idx in enumerate(tqdm(indices_analyzed, desc="Applying attention results")):
            text_content = texts_for_analysis[i]
            word_scores_dict = all_word_scores_list[i]

            top_words = self.extract_top_attention_words(word_scores_dict)
            masked_text = create_masked_sentence(text_content, top_words)

            df_to_process.at[original_idx, 'top_attention_words'] = top_words
            df_to_process.at[original_idx, self.config.TEXT_COLUMN_IN_OE_FILES] = masked_text

        return df_to_process

# === Enhanced OE Extractor ===
class MaskedTextDatasetForMetrics(TorchDataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int, is_nlp_mode: bool = False):
        self.texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_nlp_mode = is_nlp_mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            preprocess_text_for_roberta(text),
            truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0).long(),
            'attention_mask': encoding['attention_mask'].squeeze(0).long()
        }

class OEExtractorEnhanced:
    def __init__(self, config: Config, model_pl: EnhancedModel, tokenizer, device):
        self.config = config
        self.model_pl = model_pl.to(device)
        self.model_pl.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.is_nlp_mode = (config.EXPERIMENT_MODE == "nlp")

    @torch.no_grad()
    def extract_attention_metrics(self, dataloader: DataLoader, original_df_indices: Optional[pd.Index] = None,
                                  rows_to_skip_metric_calculation: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        self.model_pl.eval()
        self.model_pl.to(self.device)

        attention_metrics_list = []
        features_list = []

        print("Extracting attention metrics and features from (masked) texts...")
        for batch_encodings in tqdm(dataloader, desc="Processing (masked) text batches", leave=False):
            batch_on_device = {k: v.to(self.device) for k, v in batch_encodings.items()}

            outputs = self.model_pl.forward(batch_on_device, output_features=True, output_attentions=True)

            features_batch = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            features_list.extend(list(features_batch))

            input_ids_batch_cpu = batch_on_device['input_ids'].cpu().numpy()

            avg_attentions_batch = torch.stack([outputs.attentions[l] for l in self.config.ATTENTION_LAYER]).mean(dim=0)
            attentions_batch_layer = avg_attentions_batch.cpu().numpy()
            for i in range(len(input_ids_batch_cpu)):
                metrics = self._compute_attention_metrics_transformer(
                    attentions_batch_layer[i], input_ids_batch_cpu[i]
                )
                attention_metrics_list.append(metrics)

        metrics_df = pd.DataFrame(attention_metrics_list)
        if original_df_indices is not None:
            metrics_df.index = original_df_indices[:len(metrics_df)]

        if rows_to_skip_metric_calculation is not None:
            default_metrics_values = {'max_attention': 0.0, 'top_k_avg_attention': 0.0, 'attention_entropy': 0.0}
            for original_idx, skip in rows_to_skip_metric_calculation.items():
                if skip and original_idx in metrics_df.index:
                    for metric_name, default_val in default_metrics_values.items():
                        if metric_name in metrics_df.columns:
                             metrics_df.loc[original_idx, metric_name] = default_val

        return metrics_df, features_list

    def _compute_attention_metrics_transformer(self, attention_sample_layer, input_ids_single):
        """Transformer attention metrics 계산 - 개선된 버전"""
        try:
            # 모든 헤드의 평균 attention (CLS 토큰에서 다른 토큰으로)
            cls_attentions_to_tokens = np.mean(attention_sample_layer[:, 0, :], axis=0)
            
            # 패딩 확인을 위한 안전한 인덱싱
            max_len = min(len(input_ids_single), len(cls_attentions_to_tokens))
            
            valid_indices = []
            special_tokens = {
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                getattr(self.tokenizer, 'unk_token_id', None)
            }
            special_tokens.discard(None)  # None 값 제거
            
            for i in range(max_len):
                if input_ids_single[i] not in special_tokens:
                    valid_indices.append(i)
            
            if not valid_indices:
                return {
                    'max_attention': 0.0,
                    'top_k_avg_attention': 0.0,
                    'attention_entropy': 0.0
                }
            
            # 유효한 토큰의 attention 값만 사용
            valid_attentions = cls_attentions_to_tokens[valid_indices]
            
            # 메트릭 계산
            metrics = {
                'max_attention': float(np.max(valid_attentions)),
                'top_k_avg_attention': float(np.mean(np.sort(valid_attentions)[-min(self.config.TOP_K_ATTENTION, len(valid_attentions)):])),
                'attention_entropy': 0.0
            }
            
            # 엔트로피 계산 (안전하게)
            if len(valid_attentions) > 1:
                # Softmax 정규화
                probs = np.exp(valid_attentions - np.max(valid_attentions))
                probs = probs / (np.sum(probs) + 1e-10)
                probs = np.clip(probs, 1e-10, 1.0)
                metrics['attention_entropy'] = float(-np.sum(probs * np.log(probs)))
                
            return metrics
            
        except Exception as e:
            print(f"Warning: Error computing transformer attention metrics: {e}")
            return {'max_attention': 0.0, 'top_k_avg_attention': 0.0, 'attention_entropy': 0.0}

    def compute_removed_word_attention(self, df: pd.DataFrame, attention_analyzer: EnhancedAttentionAnalyzer,
                                       rows_to_skip_computation: Optional[pd.Series] = None) -> pd.DataFrame:
        print("Computing 'removed_avg_attention' scores...")
        df_copy = df.copy()
        if 'removed_avg_attention' not in df_copy.columns:
             df_copy['removed_avg_attention'] = 0.0

        if 'top_attention_words' not in df_copy.columns or self.config.TEXT_COLUMN not in df_copy.columns:
            print("  Required columns not found. Skipping removed_avg_attention.")
            return df_copy

        if rows_to_skip_computation is None:
            rows_to_skip_computation = pd.Series([False] * len(df_copy), index=df_copy.index)

        process_mask = ~rows_to_skip_computation
        texts_to_process = df_copy.loc[process_mask, self.config.TEXT_COLUMN].tolist()
        indices_to_process = df_copy.index[process_mask]

        if not texts_to_process:
            print("  No data to process for removed_avg_attention.")
            return df_copy

        print(f"  Getting original word attentions for {len(texts_to_process)} samples...")
        word_attentions_list_for_processing = attention_analyzer.get_word_attention_scores(texts_to_process)

        if len(word_attentions_list_for_processing) != len(texts_to_process):
            print(f"  Mismatch in lengths for removed_avg_attention. Skipping.")
            return df_copy

        print(f"  Calculating removed_avg_attention for {len(indices_to_process)} samples...")
        for i, original_idx in enumerate(tqdm(indices_to_process, desc="Calculating removed_avg_attention", leave=False)):
            top_words_val = df_copy.loc[original_idx, 'top_attention_words']
            top_words_list = safe_literal_eval(top_words_val)

            if top_words_list:
                word_scores_for_sample = word_attentions_list_for_processing[i]
                removed_scores_found = []
                for word in top_words_list:
                    score = word_scores_for_sample.get(word.lower(), word_scores_for_sample.get(word, 0.0))
                    removed_scores_found.append(score)

                if removed_scores_found:
                    df_copy.loc[original_idx, 'removed_avg_attention'] = np.mean(removed_scores_found)
                else:
                    df_copy.loc[original_idx, 'removed_avg_attention'] = 0.0
            else:
                df_copy.loc[original_idx, 'removed_avg_attention'] = 0.0

        print("'removed_avg_attention' computation complete.")
        return df_copy

    def extract_oe_datasets(self, df_with_metrics: pd.DataFrame, rows_to_exclude_from_oe: Optional[pd.Series] = None):
        print("Extracting attention-derived OE datasets...")

        if rows_to_exclude_from_oe is None:
            rows_to_exclude_from_oe = pd.Series([False] * len(df_with_metrics), index=df_with_metrics.index)

        df_for_oe_extraction = df_with_metrics[~rows_to_exclude_from_oe].copy()

        if df_for_oe_extraction.empty:
            print("No data available for OE extraction after filtering 'exclude_class'.")
            return

        print(f"Extracting OE from {len(df_for_oe_extraction)} samples.")
        print(f"Using {'Elbow Method' if self.config.USE_ELBOW_METHOD else 'Percentile Method'} for thresholding.")

        for metric_name, settings in self.config.METRIC_SETTINGS.items():
            if metric_name not in df_for_oe_extraction.columns:
                print(f"Skipping OE for metric {metric_name} - column not found.")
                continue
            self._extract_single_metric_oe(df_for_oe_extraction, metric_name, settings)

        self._extract_sequential_filtering_oe(df_for_oe_extraction)

    def _extract_single_metric_oe(self, df: pd.DataFrame, metric: str, settings: dict):
        scores = np.nan_to_num(df[metric].values, nan=0.0, posinf=0.0, neginf=0.0)

        if len(scores) < 3:
            print(f"Not enough scores for metric {metric} to extract OE. Skipping.")
            return

        mode_desc = ""
        selected_indices = np.array([])

        if self.config.USE_ELBOW_METHOD:
            sorted_indices = np.argsort(scores)
            elbow_idx_in_sorted = find_elbow_point(scores)

            if settings['mode'] == 'higher':
                selected_original_indices = sorted_indices[elbow_idx_in_sorted:]
            else:
                selected_original_indices = sorted_indices[:elbow_idx_in_sorted]

            selected_indices = df.index.get_indexer(df.index[selected_original_indices])
            mode_desc = f"elbow_{settings['mode']}"
        else:
            if settings['mode'] == 'higher':
                threshold = np.percentile(scores, 100 - settings['percentile'])
                selected_indices = np.where(scores >= threshold)[0]
            else:
                threshold = np.percentile(scores, settings['percentile'])
                selected_indices = np.where(scores <= threshold)[0]
            mode_desc = f"{settings['mode']}{settings['percentile']}pct"

        if len(selected_indices) > 0:
            oe_df_simple = df.iloc[selected_indices][[self.config.TEXT_COLUMN_IN_OE_FILES]].copy()
            extended_cols = [self.config.TEXT_COLUMN_IN_OE_FILES, self.config.TEXT_COLUMN, 'top_attention_words', metric]
            
            # removed_avg_attention 추가
            if 'removed_avg_attention' in df.columns:
                extended_cols.append('removed_avg_attention')
            
            label_col_name = 'label' if self.is_nlp_mode else self.config.CLASS_COLUMN
            if label_col_name in df.columns: extended_cols.append(label_col_name)

            extended_cols_present = [col for col in extended_cols if col in df.columns]
            oe_df_extended = df.iloc[selected_indices][extended_cols_present].copy()

            oe_filename_base = f"oe_data_{metric}_{mode_desc}"

            simple_path = os.path.join(self.config.OE_DATA_DIR, f"{oe_filename_base}.csv")
            extended_path = os.path.join(self.config.OE_DATA_DIR, f"{oe_filename_base}_extended.csv")

            oe_df_simple.to_csv(simple_path, index=False)
            oe_df_extended.to_csv(extended_path, index=False)
            print(f"Saved OE dataset ({len(oe_df_simple)} samples) for {metric} {mode_desc} to {simple_path}")
        else:
            print(f"No samples selected for OE with metric {metric} and settings {settings}")

    def _extract_sequential_filtering_oe(self, df: pd.DataFrame):
        print("Applying sequential filtering for OE extraction...")
        current_selection_df = df.copy()

        filter_desc_parts = []
        for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
            if metric not in current_selection_df.columns:
                print(f"Seq Filter Step {step+1}: Metric '{metric}' not found. Skipping.")
                continue

            if current_selection_df.empty:
                print(f"No samples left before filter: {metric}. Stopping.")
                break

            scores = np.nan_to_num(current_selection_df[metric].values, nan=0.0, posinf=0.0, neginf=0.0)

            if len(scores) < 3:
                print(f"Not enough scores for sequential filter on {metric}. Stopping.")
                break

            step_mask = np.zeros(len(current_selection_df), dtype=bool)

            if self.config.USE_ELBOW_METHOD:
                sorted_indices_local = np.argsort(scores)
                elbow_idx_in_sorted_local = find_elbow_point(scores)

                if settings['mode'] == 'higher':
                    selected_local_indices = sorted_indices_local[elbow_idx_in_sorted_local:]
                else:
                    selected_local_indices = sorted_indices_local[:elbow_idx_in_sorted_local]

                step_mask[selected_local_indices] = True
                filter_desc_parts.append(f"{metric}_elbow_{settings['mode']}")
            else:
                if settings['mode'] == 'higher':
                    threshold = np.percentile(scores, 100 - settings['percentile'])
                    step_mask = scores >= threshold
                else:
                    threshold = np.percentile(scores, settings['percentile'])
                    step_mask = scores <= threshold
                filter_desc_parts.append(f"{metric}_{settings['mode']}{settings['percentile']}")

            current_selection_df = current_selection_df[step_mask]
            print(f"Seq Filter {step+1} ({metric} {'elbow' if self.config.USE_ELBOW_METHOD else 'percentile'}): {len(current_selection_df)} samples remaining")

        if not current_selection_df.empty:
            oe_df_simple = current_selection_df[[self.config.TEXT_COLUMN_IN_OE_FILES]].copy()
            extended_cols = [self.config.TEXT_COLUMN_IN_OE_FILES, self.config.TEXT_COLUMN, 'top_attention_words']
            metrics_in_seq = [m_name for m_name, _ in self.config.FILTERING_SEQUENCE if m_name in current_selection_df.columns]
            extended_cols.extend(metrics_in_seq)
            
            # removed_avg_attention 추가
            if 'removed_avg_attention' in current_selection_df.columns:
                extended_cols.append('removed_avg_attention')
                
            label_col_name = 'label' if self.is_nlp_mode else self.config.CLASS_COLUMN
            if label_col_name in current_selection_df.columns: extended_cols.append(label_col_name)

            extended_cols_present = [col for col in extended_cols if col in current_selection_df.columns]
            oe_df_extended = current_selection_df[extended_cols_present].copy()

            filter_desc_str = "_".join(filter_desc_parts)
            oe_filename_base = f"oe_data_sequential_{filter_desc_str}"

            simple_path = os.path.join(self.config.OE_DATA_DIR, f"{oe_filename_base}.csv")
            extended_path = os.path.join(self.config.OE_DATA_DIR, f"{oe_filename_base}_extended.csv")

            oe_df_simple.to_csv(simple_path, index=False)
            oe_df_extended.to_csv(extended_path, index=False)
            print(f"Saved sequential OE dataset ({len(oe_df_simple)} samples) to {simple_path}")
        else:
            print("No samples selected by sequential filtering.")

# === Enhanced Visualizer ===
class EnhancedVisualizer:
    def __init__(self, config: Config):
        self.config = config
        self.is_nlp_mode = (config.EXPERIMENT_MODE == "nlp")

    def plot_metric_distribution(self, scores: np.ndarray, metric_name: str, title: str, save_path: str):
        if len(scores) == 0:
            print(f"No scores for metric {metric_name}. Skipping plot.")
            return
        plt.figure(figsize=(10, 6))
        valid_scores = scores[np.isfinite(scores)]
        if len(valid_scores) < 2 or len(np.unique(valid_scores)) < 2:
             plt.hist(valid_scores, bins=min(50, max(1, len(valid_scores))), density=False, alpha=0.7, label='Histogram (counts)')
             plt.ylabel('Count', fontsize=12)
             kde_available = False
        elif SNS_AVAILABLE:
            sns.histplot(valid_scores, bins=50, kde=True, stat='density')
            plt.ylabel('Density', fontsize=12)
            kde_available = True
        else:
            plt.hist(valid_scores, bins=50, density=True, alpha=0.7)
            plt.ylabel('Density', fontsize=12)
            kde_available = False

        plt.title(title, fontsize=14)
        plt.xlabel(metric_name, fontsize=12)
        plt.grid(alpha=0.3)
        if len(valid_scores) > 0:
            mean_val = np.mean(valid_scores)
            plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        if kde_available or len(valid_scores) > 0:
            print(f"Distribution plot saved: {save_path}")
        else:
            print(f"Distribution plot for {metric_name} skipped due to insufficient data.")

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, title: str, save_path: str,
                  highlight_indices: Optional[np.ndarray] = None, highlight_label: str = 'OE Candidate',
                  class_names: Optional[Dict] = None, seed: int = 42):
        if features is None or len(features) == 0:
            print(f"No features for t-SNE plot: {title}. Skipping.")
            return

        if isinstance(features, list): features = np.array(features)
        if features.ndim == 1: features = features.reshape(-1, 1)
        if features.shape[0] != len(labels):
            print(f"Feature ({features.shape[0]}) and label ({len(labels)}) length mismatch for t-SNE. Skipping.")
            return

        print(f"Running t-SNE for '{title}' on {features.shape[0]} samples...")

        perplexity_val = min(30.0, float(max(0, features.shape[0] - 1)))
        if perplexity_val <= 1.0:
            if features.shape[0] < 5:
                print("Too few samples for meaningful t-SNE. Skipping.")
                return
            perplexity_val = min(5.0, float(max(0, features.shape[0] - 1))) if features.shape[0] > 1 else 0

        if perplexity_val == 0 and features.shape[0] == 1:
             tsne_results = np.array([[0,0]])
        elif features.shape[0] <= perplexity_val:
             perplexity_val = max(1.0, features.shape[0] - 1.0)
             if perplexity_val <= 1 and features.shape[0] > 1: perplexity_val = float(features.shape[0]-1) / 2
             if perplexity_val == 0:
                print("Cannot run TSNE with these dimensions. Skipping.")
                return
             print(f"Adjusted perplexity to {perplexity_val} due to small sample size.")
             tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity_val, init='pca', learning_rate='auto', max_iter=250)
             try: tsne_results = tsne.fit_transform(features)
             except Exception as e: print(f"Error during t-SNE: {e}. Skipping plot."); return
        else:
            tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity_val, init='pca', learning_rate='auto', max_iter=max(250,min(1000, features.shape[0]*2)))
            try: tsne_results = tsne.fit_transform(features)
            except Exception as e: print(f"Error during t-SNE: {e}. Skipping plot."); return

        df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        df_tsne['label_val'] = labels
        df_tsne['is_highlighted'] = False
        if highlight_indices is not None and len(highlight_indices) > 0:
            valid_highlight_indices = [h_idx for h_idx in highlight_indices if 0 <= h_idx < len(df_tsne)]
            if valid_highlight_indices:
                 df_tsne.loc[valid_highlight_indices, 'is_highlighted'] = True

        plt.figure(figsize=(14, 10))
        unique_label_vals = sorted(df_tsne['label_val'].unique())

        if len(unique_label_vals) == 0: plt.close(); return
        palette = sns.color_palette("husl", len(unique_label_vals)) if SNS_AVAILABLE else plt.cm.get_cmap('tab20', len(unique_label_vals))

        for i, label_val_item in enumerate(unique_label_vals):
            subset = df_tsne[(df_tsne['label_val'] == label_val_item) & (~df_tsne['is_highlighted'])]
            if not subset.empty:
                c_name = class_names.get(label_val_item, f'Class {label_val_item}') if class_names else f'Label {label_val_item}'
                color_val = palette(i) if callable(palette) else palette[i % len(palette)]
                plt.scatter(subset['tsne1'], subset['tsne2'], color=color_val, label=c_name, alpha=0.7, s=30)

        highlighted_subset = df_tsne[df_tsne['is_highlighted']]
        if not highlighted_subset.empty:
            plt.scatter(highlighted_subset['tsne1'], highlighted_subset['tsne2'],
                        color='red', marker='x', s=100, label=highlight_label, alpha=0.9, zorder=5)

        plt.title(title, fontsize=16, pad=20)
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')

        if len(unique_label_vals) + (1 if not highlighted_subset.empty else 0) > 15:
            plt.legend(loc='best', fontsize=10, frameon=True)
        else:
            plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True, fancybox=True)
            plt.subplots_adjust(right=0.78)

        plt.tight_layout(rect=[0, 0, 0.9 if len(unique_label_vals) <=15 else 1, 1])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE plot saved: {save_path}")

    def visualize_all_metrics(self, df: pd.DataFrame):
        metric_columns = ['max_attention', 'top_k_avg_attention', 'attention_entropy', 'removed_avg_attention']
        for metric in metric_columns:
            if metric in df.columns and not df[metric].isnull().all():
                self.plot_metric_distribution(
                    df[metric].dropna().values, metric, f'Distribution of {metric}',
                    os.path.join(self.config.VIS_DIR, f'{metric}_distribution_{self.config.EXPERIMENT_MODE}.png')
                )

    def visualize_oe_candidates(self, df: pd.DataFrame, features_list: List[np.ndarray],
                                label2id: dict, id2label: dict):
        if not features_list or len(features_list) != len(df):
            print(f"Features list mismatch with DataFrame or empty. Skipping t-SNE for OE.")
            return

        features_np = np.array(features_list)

        tsne_labels = []
        if self.is_nlp_mode:
            for label_id_val in df['label_id']:
                tsne_labels.append(int(label_id_val))
        else:
            unknown_class_indicator, other_filtered_indicator = -1, -2
            exclude_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
            for cls_val_str in df[self.config.CLASS_COLUMN].astype(str):
                cls_lower = cls_val_str.lower()
                if cls_lower == exclude_class_lower:
                    tsne_labels.append(unknown_class_indicator)
                else:
                    tsne_labels.append(label2id.get(cls_lower, other_filtered_indicator))

        tsne_labels_np = np.array(tsne_labels)

        class_names_viz = {**{id_val: str(name) for id_val, name in id2label.items()}}
        if self.is_nlp_mode:
            class_names_viz[-1] = 'Other ID (Not in Train)'
        else:
            class_names_viz[-1] = f'Unknown ({self.config.EXCLUDE_CLASS_FOR_TRAINING})'
            class_names_viz[-2] = 'Other/Filtered ID'

        for metric, settings in self.config.METRIC_SETTINGS.items():
            if metric not in df.columns: continue

            scores = np.nan_to_num(df[metric].values, nan=0.0, posinf=0.0, neginf=0.0)
            if len(scores) < 3: continue

            oe_indices = np.array([])
            mode_desc = ""
            if self.config.USE_ELBOW_METHOD:
                sorted_indices = np.argsort(scores)
                elbow_idx_in_sorted = find_elbow_point(scores)
                if settings['mode'] == 'higher':
                    oe_indices_original = sorted_indices[elbow_idx_in_sorted:]
                else:
                    oe_indices_original = sorted_indices[:elbow_idx_in_sorted]
                oe_indices = df.index.get_indexer(df.index[oe_indices_original])
                mode_desc = f"elbow_{settings['mode']}"
            else:
                if settings['mode'] == 'higher':
                    threshold = np.percentile(scores, 100 - settings['percentile'])
                    oe_indices = np.where(scores >= threshold)[0]
                else:
                    threshold = np.percentile(scores, settings['percentile'])
                    oe_indices = np.where(scores <= threshold)[0]
                mode_desc = f"{settings['mode']}{settings['percentile']}%"

            plot_title = f't-SNE: OE by {metric} ({mode_desc}) ({self.config.EXPERIMENT_MODE})'
            save_name = f'tsne_oe_cand_{metric}_{mode_desc}_{self.config.EXPERIMENT_MODE}.png'

            self.plot_tsne(
                features_np, tsne_labels_np, plot_title,
                os.path.join(self.config.VIS_DIR, save_name),
                highlight_indices=oe_indices, highlight_label=f'OE ({metric} {mode_desc})',
                class_names=class_names_viz, seed=self.config.RANDOM_STATE
            )

        if hasattr(self.config, 'FILTERING_SEQUENCE') and self.config.FILTERING_SEQUENCE:
            current_selection_df_indices = df.index.to_series()
            filter_steps_desc_list = []
            for step, (metric, settings) in enumerate(self.config.FILTERING_SEQUENCE):
                if metric not in df.columns: continue
                if current_selection_df_indices.empty: break

                scores_subset = np.nan_to_num(df.loc[current_selection_df_indices, metric].values, nan=0.0, posinf=0.0, neginf=0.0)
                if len(scores_subset) < 3: break

                step_mask_on_subset = np.zeros(len(scores_subset), dtype=bool)
                if self.config.USE_ELBOW_METHOD:
                    sorted_indices_local = np.argsort(scores_subset)
                    elbow_idx_local = find_elbow_point(scores_subset)
                    if settings['mode'] == 'higher':
                        selected_local_indices = sorted_indices_local[elbow_idx_local:]
                    else:
                        selected_local_indices = sorted_indices_local[:elbow_idx_local]
                    step_mask_on_subset[selected_local_indices] = True
                    filter_steps_desc_list.append(f"{metric[0:3]}_elbow_{settings['mode'][0]}")
                else:
                    if settings['mode'] == 'higher':
                        threshold = np.percentile(scores_subset, 100 - settings['percentile'])
                        step_mask_on_subset = scores_subset >= threshold
                    else:
                        threshold = np.percentile(scores_subset, settings['percentile'])
                        step_mask_on_subset = scores_subset <= threshold
                    filter_steps_desc_list.append(f"{metric[0:3]}{settings['percentile']}{settings['mode'][0]}")

                current_selection_df_indices = current_selection_df_indices[step_mask_on_subset]

            final_indices_seq_original = current_selection_df_indices.index
            final_indices_seq = df.index.get_indexer(final_indices_seq_original)

            if len(final_indices_seq) > 0:
                seq_desc_short = "_to_".join(filter_steps_desc_list)
                plot_title_seq = f't-SNE: Sequential OE ({seq_desc_short}) ({self.config.EXPERIMENT_MODE})'
                save_name_seq = f'tsne_oe_cand_sequential_{seq_desc_short}_{self.config.EXPERIMENT_MODE}.png'

                self.plot_tsne(
                    features_np, tsne_labels_np, plot_title_seq,
                    os.path.join(self.config.VIS_DIR, save_name_seq),
                    highlight_indices=final_indices_seq,
                    highlight_label=f'Sequential OE ({len(final_indices_seq)} samples)',
                    class_names=class_names_viz, seed=self.config.RANDOM_STATE
                )

# ==============================================================================
# === OSR Components (NLP and Syslog) ===
# ==============================================================================

# OSR용 OOD 모델
class RoBERTaOOD(nn.Module):
    def __init__(self, model_name: str, num_labels: int, cache_dir: Optional[str] = None):
        super().__init__()
        print(f"Initializing RoBERTaOOD with {num_labels} classes. Model: {model_name}")
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            cache_dir=cache_dir
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_features: bool = False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if output_features:
            features = outputs.hidden_states[-1][:, 0, :]
            return logits, features
        return logits

# OSR Lightning Module
class OSRLightningModule(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int, learning_rate: float, 
                 oe_lambda: float = 1.0, temperature: float = 1.0, 
                 cache_dir: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = RoBERTaOOD(model_name, num_labels, cache_dir)
        self.learning_rate = learning_rate
        self.oe_lambda = oe_lambda
        self.temperature = temperature
        self.num_labels = num_labels
        
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)
        
    def forward(self, input_ids, attention_mask, output_features=False):
        return self.model(input_ids, attention_mask, output_features)
    
    def training_step(self, batch, batch_idx):
        # ID와 OE 데이터를 분리
        if isinstance(batch, dict):
            # Single batch (ID only)
            id_batch = batch
            oe_batch = None
        else:
            # Combined batch (ID, OE)
            id_batch, oe_batch = batch
            
        # ID loss
        id_logits = self.model(id_batch['input_ids'], id_batch['attention_mask'])
        id_loss = F.cross_entropy(id_logits, id_batch['label'])
        
        # ID accuracy
        preds = torch.argmax(id_logits, dim=1)
        self.train_accuracy.update(preds, id_batch['label'])
        
        # OE loss (if available)
        oe_loss = torch.tensor(0.0).to(self.device)
        if oe_batch is not None:
            oe_logits = self.model(oe_batch['input_ids'], oe_batch['attention_mask'])
            uniform_target = torch.full_like(oe_logits, 1.0 / self.num_labels)
            oe_loss = F.kl_div(
                F.log_softmax(oe_logits / self.temperature, dim=1), 
                uniform_target, 
                reduction='batchmean', 
                log_target=False
            )
        
        # Combined loss
        total_loss = id_loss + self.oe_lambda * oe_loss
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_id_loss', id_loss, on_step=False, on_epoch=True)
        self.log('train_oe_loss', oe_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.model(batch['input_ids'], batch['attention_mask'])
        loss = F.cross_entropy(logits, batch['label'])
        
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, batch['label'])
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        train_acc = self.train_accuracy.compute()
        self.log('train_accuracy', train_acc, prog_bar=True)
        self.train_accuracy.reset()
        
    def on_validation_epoch_end(self):
        val_acc = self.val_accuracy.compute()
        self.log('val_accuracy', val_acc, prog_bar=True)
        self.val_accuracy.reset()
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

# Combined DataLoader for OSR training
class CombinedOSRDataLoader:
    def __init__(self, id_loader, oe_loader=None):
        self.id_loader = id_loader
        self.oe_loader = oe_loader
        
    def __len__(self):
        return len(self.id_loader)
    
    def __iter__(self):
        id_iter = iter(self.id_loader)
        oe_iter = iter(self.oe_loader) if self.oe_loader else None
        
        for id_batch in id_iter:
            if oe_iter:
                try:
                    oe_batch = next(oe_iter)
                except StopIteration:
                    oe_iter = iter(self.oe_loader)
                    oe_batch = next(oe_iter)
                yield id_batch, oe_batch
            else:
                yield id_batch

# 데이터 준비 함수들
def prepare_nlp_id_data_for_osr(datamodule: EnhancedDataModule, tokenizer, max_length: int):
    """Prepares In-Distribution (ID) train and test datasets for NLP OSR."""
    print(f"\n--- Preparing NLP ID data for OSR ({datamodule.config.CURRENT_NLP_DATASET}) ---")
    if datamodule.train_df_final is None:
        print("Error: NLP DataModule not set up.")
        return None, None, 0, {}, {}

    id_train_texts = datamodule.train_df_final['text'].tolist()
    id_train_labels = datamodule.train_df_final['label_id'].tolist()

    id_test_texts = datamodule.val_df_final['text'].tolist()
    id_test_labels = datamodule.val_df_final['label_id'].tolist()

    num_classes = datamodule.num_labels
    id_label2id = datamodule.label2id
    id_id2label = datamodule.id2label

    train_dataset = OSRNLPDataset(id_train_texts, id_train_labels, tokenizer, max_length)
    id_test_dataset = OSRNLPDataset(id_test_texts, id_test_labels, tokenizer, max_length)

    print(f"  - NLP OSR ID Train: {len(train_dataset)} samples, ID Test: {len(id_test_dataset)} samples.")
    return train_dataset, id_test_dataset, num_classes, id_label2id, id_id2label

def prepare_syslog_id_data_for_osr(datamodule: EnhancedDataModule, tokenizer, max_length: int):
    """Prepares In-Distribution (ID) train and test datasets for Syslog OSR."""
    print("\n--- Preparing Syslog ID data for OSR ---")
    if datamodule.train_df_final is None:
        print("Error: Syslog DataModule not set up.")
        return None, None, 0, {}, {}

    id_train_texts = datamodule.train_df_final[Config.TEXT_COLUMN].tolist()
    id_train_labels = datamodule.train_df_final['label'].tolist()

    id_test_texts = datamodule.val_df_final[Config.TEXT_COLUMN].tolist()
    id_test_labels = datamodule.val_df_final['label'].tolist()

    num_classes = datamodule.num_labels
    id_label2id = datamodule.label2id
    id_id2label = datamodule.id2label

    train_dataset = OSRSyslogTextDataset(id_train_texts, id_train_labels, tokenizer, max_length)
    id_test_dataset = OSRSyslogTextDataset(id_test_texts, id_test_labels, tokenizer, max_length)

    print(f"  - Syslog OSR ID Train: {len(train_dataset)} samples, ID Test: {len(id_test_dataset)} samples.")
    return train_dataset, id_test_dataset, num_classes, id_label2id, id_id2label

def prepare_syslog_unknown_as_ood(tokenizer, max_length: int) -> Optional[OSRSyslogTextDataset]:
    """Loads Syslog 'unknown' data and prepares it as an OOD test set."""
    print("\n--- Preparing Syslog 'unknown' class as OOD dataset ---")
    data_path = Config.ORIGINAL_DATA_PATH
    if not os.path.exists(data_path):
        print(f"  - Syslog data file not found: {data_path}. Skipping.")
        return None

    df_full = safe_load_dataset_file(data_path)
    if df_full is None:
        print(f"  - Failed to load syslog data from {data_path}. Skipping.")
        return None
        
    # 'unknown' 클래스를 OOD로 사용
    ood_df = df_full[df_full[Config.CLASS_COLUMN].astype(str).str.lower() == Config.EXCLUDE_CLASS_FOR_TRAINING.lower()]
    if ood_df.empty:
        print(f"  - No 'unknown' samples found in syslog data. Skipping.")
        return None

    texts = ood_df[Config.TEXT_COLUMN].tolist()
    labels = [-1] * len(texts)
    ood_dataset = OSRSyslogTextDataset(texts, labels, tokenizer, max_length)
    print(f"  - Loaded {len(ood_dataset)} 'unknown' syslog samples as OOD.")
    return ood_dataset

def prepare_syslog_as_ood_for_nlp(tokenizer, max_length: int) -> Optional[OSRNLPDataset]:
    """Loads Syslog 'unknown' data and prepares it as an OOD test set for an NLP model."""
    print("\n--- Preparing Syslog dataset as OOD for NLP model ---")
    data_path = Config.ORIGINAL_DATA_PATH
    if not os.path.exists(data_path):
        print(f"  - Syslog data file not found: {data_path}. Skipping.")
        return None

    df_full = safe_load_dataset_file(data_path)
    if df_full is None:
        print(f"  - Failed to load syslog data from {data_path}. Skipping.")
        return None
        
    # 'unknown' 클래스를 OOD로 사용
    ood_df = df_full[df_full[Config.CLASS_COLUMN].astype(str).str.lower() == Config.EXCLUDE_CLASS_FOR_TRAINING.lower()]
    if ood_df.empty:
        print(f"  - No 'unknown' samples found in syslog data. Skipping.")
        return None

    texts = ood_df[Config.TEXT_COLUMN].tolist()
    labels = [-1] * len(texts)
    ood_dataset = OSRNLPDataset(texts, labels, tokenizer, max_length)
    print(f"  - Loaded {len(ood_dataset)} 'unknown' syslog samples as OOD.")
    return ood_dataset

def prepare_nlp_as_ood_for_syslog(dataset_key: str, tokenizer, max_length: int) -> Optional[OSRSyslogTextDataset]:
    """Loads an NLP dataset and prepares it as an OOD test set for a Syslog model."""
    print(f"\n--- Preparing NLP dataset '{dataset_key}' as OOD for Syslog model ---")
    data_dict = NLPDatasetLoader.load_any_dataset(dataset_key, split='test')

    if not data_dict or not data_dict.get('text'):
        print(f"  - Failed to load or no text found in '{dataset_key}'.")
        return None

    all_texts = data_dict['text']
    labels = [-1] * len(all_texts)
    ood_dataset = OSRSyslogTextDataset(all_texts, labels, tokenizer, max_length)
    print(f"  - Loaded {len(ood_dataset)} samples from '{dataset_key}' as OOD.")
    return ood_dataset

def prepare_attention_derived_oe_data_for_osr(tokenizer, max_length: int, oe_data_path: str, oe_text_col: str, for_syslog: bool = False) -> Optional[Union[OSRNLPDataset, OSRSyslogTextDataset]]:
    print(f"\n--- Preparing Attention-Derived OE data from: {oe_data_path} ---")
    if not os.path.exists(oe_data_path): return None
    try:
        df = pd.read_csv(oe_data_path)
        if oe_text_col not in df.columns:
            fallback_cols = ['masked_text_attention', 'text', Config.TEXT_COLUMN]
            found_col = False
            for col_attempt in fallback_cols:
                if col_attempt in df.columns:
                    oe_text_col_actual = col_attempt
                    found_col = True
                    break
            if not found_col: raise ValueError(f"OE CSV must contain a valid text column.")
        else:
            oe_text_col_actual = oe_text_col

        texts = df.dropna(subset=[oe_text_col_actual])[oe_text_col_actual].astype(str).tolist()
        if not texts: return None

        oe_labels = [-1] * len(texts)
        DatasetClass = OSRSyslogTextDataset if for_syslog else OSRNLPDataset
        oe_dataset = DatasetClass(texts, oe_labels, tokenizer, max_length)
        print(f"  - Loaded {len(oe_dataset)} samples from {oe_data_path}.")
        return oe_dataset
    except Exception as e:
        print(f"Error preparing attention-derived OE data: {e}")
        return None

# === 일반화된 OSR 평가 함수 ===
def evaluate_osr(model, id_loader, ood_loader, device, temperature=1.0, threshold_percentile=5.0, return_data=False, mode="syslog"):
    """Evaluates an OSR model (generalized for NLP and Syslog)."""
    model.eval()
    model.to(device)

    all_id_scores, all_id_labels_true, all_id_labels_pred, all_id_features = [], [], [], []
    all_ood_scores, all_ood_features = [], []

    with torch.no_grad():
        for batch in tqdm(id_loader, desc=f"Evaluating ID ({mode.upper()})", leave=False):
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels_true = batch['label']
            logits, features = model(input_ids, attention_mask, output_features=True)

            softmax_probs = F.softmax(logits / temperature, dim=1)
            max_probs, preds_id = softmax_probs.max(dim=1)

            all_id_scores.append(max_probs.cpu())
            all_id_labels_true.extend(labels_true.numpy())
            all_id_labels_pred.extend(preds_id.cpu().numpy())
            all_id_features.append(features.cpu())

    if ood_loader:
        with torch.no_grad():
            for batch in tqdm(ood_loader, desc=f"Evaluating OOD ({mode.upper()})", leave=False):
                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                logits, features = model(input_ids, attention_mask, output_features=True)

                softmax_probs = F.softmax(logits / temperature, dim=1)
                max_probs, _ = softmax_probs.max(dim=1)

                all_ood_scores.append(max_probs.cpu())
                all_ood_features.append(features.cpu())

    id_scores_np = torch.cat(all_id_scores).numpy() if all_id_scores else np.array([])
    id_features_np = torch.cat(all_id_features).numpy() if all_id_features else np.array([])
    id_labels_true_np = np.array(all_id_labels_true)
    id_labels_pred_np = np.array(all_id_labels_pred)
    ood_scores_np = torch.cat(all_ood_scores).numpy() if all_ood_scores else np.array([])
    ood_features_np = torch.cat(all_ood_features).numpy() if all_ood_features else np.array([])

    results = {
        "Closed_Set_Accuracy": 0.0, "F1_Macro": 0.0, "AUROC_OOD": 0.0,
        "FPR@TPR95": 1.0, "AUPR_In": 0.0, "AUPR_Out": 0.0, "DetectionAccuracy": 0.0,
        "OSCR": 0.0, "Threshold_Used": 0.0
    }
    plot_data = {"id_scores": id_scores_np, "ood_scores": ood_scores_np,
                 "id_labels_true": id_labels_true_np, "id_labels_pred": id_labels_pred_np,
                 "id_features": id_features_np, "ood_features": ood_features_np}

    if len(id_labels_true_np) == 0: return (results, plot_data) if return_data else results

    results["Closed_Set_Accuracy"] = accuracy_score(id_labels_true_np, id_labels_pred_np)
    results["F1_Macro"] = f1_score(id_labels_true_np, id_labels_pred_np, average='macro', zero_division=0)

    if len(ood_scores_np) == 0: return (results, plot_data) if return_data else results

    y_true = np.concatenate([np.ones_like(id_scores_np), np.zeros_like(ood_scores_np)])
    y_scores = np.concatenate([id_scores_np, ood_scores_np])

    if len(np.unique(y_true)) > 1:
        results["AUROC_OOD"] = roc_auc_score(y_true, y_scores)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        idx_tpr95 = np.where(tpr >= 0.95)[0]
        results["FPR@TPR95"] = fpr[idx_tpr95[0]] if len(idx_tpr95) > 0 else 1.0
        prec_in, recall_in, _ = precision_recall_curve(y_true, y_scores, pos_label=1)
        results["AUPR_In"] = auc(recall_in, prec_in)
        prec_out, recall_out, _ = precision_recall_curve(1 - y_true, 1 - y_scores, pos_label=1)
        results["AUPR_Out"] = auc(recall_out, prec_out)

    threshold = np.percentile(id_scores_np, threshold_percentile)
    results["Threshold_Used"] = float(threshold)

    id_preds_binary = (id_scores_np >= threshold).astype(int)
    ood_preds_binary = (ood_scores_np < threshold).astype(int)
    results["DetectionAccuracy"] = (np.sum(id_preds_binary) + np.sum(ood_preds_binary)) / (len(id_scores_np) + len(ood_scores_np))

    known_mask = id_scores_np >= threshold
    ccr = accuracy_score(id_labels_true_np[known_mask], id_labels_pred_np[known_mask]) if np.sum(known_mask) > 0 else 0.0
    oer = np.sum(ood_scores_np >= threshold) / len(ood_scores_np) if len(ood_scores_np) > 0 else 0.0
    results["OSCR"] = ccr * (1.0 - oer)

    return (results, plot_data) if return_data else results

# OSR 플롯 함수들
def plot_confidence_histograms_osr(id_scores, ood_scores, title, save_path):
    plt.figure(figsize=(10, 6))
    if SNS_AVAILABLE:
        sns.histplot(id_scores, bins=50, color='blue', label='ID Confidence', kde=True, stat='density', alpha=0.6)
        sns.histplot(ood_scores, bins=50, color='red', label='OOD Confidence', kde=True, stat='density', alpha=0.6)
    else:
        plt.hist(id_scores, bins=50, color='blue', label='ID Confidence', density=True, alpha=0.6)
        plt.hist(ood_scores, bins=50, color='red', label='OOD Confidence', density=True, alpha=0.6)
    plt.title(title, fontsize=16)
    plt.xlabel('Max Softmax Probability (Confidence)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve_osr(id_scores, ood_scores, title, save_path):
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix_osr(cm, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    if SNS_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names, rotation=0)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tsne_osr(id_features, ood_features, title, save_path, seed=42):
    if len(id_features) == 0 and len(ood_features) == 0:
        print(f"No features for t-SNE plot '{title}'. Skipping.")
        return

    features = np.concatenate([id_features, ood_features], axis=0)
    labels = np.concatenate([np.ones(len(id_features)), np.zeros(len(ood_features))])

    perplexity_val = min(30.0, float(max(1, features.shape[0] - 1)))
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=seed, init='pca', learning_rate='auto')

    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[labels==1, 0], tsne_results[labels==1, 1], c='blue', label='ID', alpha=0.5)
    plt.scatter(tsne_results[labels==0, 0], tsne_results[labels==0, 1], c='red', label='OOD', alpha=0.5)
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE dim 1", fontsize=12)
    plt.ylabel("t-SNE dim 2", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

# === 종합 NLP 데이터셋 관리 클래스 ===
class ComprehensiveNLPManager:
    """모든 NLP 데이터셋을 각각 ID로 사용하여 OSR 실험을 수행하는 관리 클래스"""
    def __init__(self, config: Config):
        self.config = config
        self.nlp_datasets_for_experiments = config.NLP_DATASETS_FOR_COMPREHENSIVE
        self.original_current_dataset = config.CURRENT_NLP_DATASET
        self.original_paths = self._save_original_paths()

    def _save_original_paths(self) -> Dict[str, str]:
        """원래 경로들을 저장"""
        return {
            'OUTPUT_DIR': self.config.OUTPUT_DIR,
            'MODEL_SAVE_DIR': self.config.MODEL_SAVE_DIR,
            'LOG_DIR': self.config.LOG_DIR,
            'CONFUSION_MATRIX_DIR': self.config.CONFUSION_MATRIX_DIR,
            'VIS_DIR': self.config.VIS_DIR,
            'OE_DATA_DIR': self.config.OE_DATA_DIR,
            'ATTENTION_DATA_DIR': self.config.ATTENTION_DATA_DIR,
            'OSR_EXPERIMENT_DIR': self.config.OSR_EXPERIMENT_DIR,
            'OSR_MODEL_DIR': self.config.OSR_MODEL_DIR,
            'OSR_RESULT_DIR': self.config.OSR_RESULT_DIR,
        }

    def get_datasets_for_comprehensive_experiments(self) -> List[str]:
        """종합 실험에 사용할 데이터셋 목록 반환"""
        return self.nlp_datasets_for_experiments

    def setup_config_for_dataset(self, dataset_name: str):
        """특정 데이터셋에 맞게 Config 설정 업데이트"""
        self.config.CURRENT_NLP_DATASET = dataset_name
        self.config.update_paths_for_dataset(dataset_name, self.config.BASE_OUTPUT_DIR)
        self.config.create_directories()

    def restore_original_config(self):
        """원래 Config 설정으로 복원"""
        self.config.CURRENT_NLP_DATASET = self.original_current_dataset
        for key, value in self.original_paths.items():
            setattr(self.config, key, value)

# === Main Pipeline Class ===
class EnhancedOEPipeline:
    def __init__(self, config: Config):
        self.config = config
        config.create_directories()
        config.save_config()
        set_seed(config.RANDOM_STATE)

        self.data_module: Optional[EnhancedDataModule] = None
        self.model: Optional[EnhancedModel] = None
        self.attention_analyzer: Optional[EnhancedAttentionAnalyzer] = None
        self.oe_extractor: Optional[OEExtractorEnhanced] = None
        self.visualizer = EnhancedVisualizer(config)
        
        # 종합 NLP 실험 관리자
        self.comprehensive_nlp_manager = ComprehensiveNLPManager(config) if config.EXPERIMENT_MODE == "nlp" else None

    def _reset_pipeline_state(self):
        """파이프라인 상태를 완전히 초기화"""
        self.data_module = None
        self.model = None
        self.attention_analyzer = None
        self.oe_extractor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _cleanup_after_experiment(self):
        """실험 후 메모리 정리"""
        self._reset_pipeline_state()
        clear_memory()

    def run_stage1_model_training(self):
        if not self.config.STAGE_MODEL_TRAINING:
            print("Skipping Stage 1: Base Model Training.")
            if self._check_existing_model(): 
                self._load_existing_model()
            else: 
                print("Error: Model training skipped, but no existing model found.")
            return

        print(f"\n{'='*50}\nSTAGE 1: BASE MODEL TRAINING ({self.config.EXPERIMENT_MODE.upper()})\n{'='*50}")
        self.data_module = EnhancedDataModule(self.config)
        self.data_module.prepare_data()
        self.data_module.setup()

        self.model = EnhancedModel(
            config=self.config, 
            num_labels=self.data_module.num_labels,
            label2id=self.data_module.label2id, 
            id2label=self.data_module.id2label,
            class_weights=self.data_module.class_weights
        )

        # Validation 데이터 존재 확인
        has_validation = hasattr(self.data_module, 'val_df_final') and len(self.data_module.val_df_final) > 0
        
        if has_validation:
            monitor_metric = 'val_f1_macro'
            monitor_mode = 'max'
        else:
            monitor_metric = 'train_f1_macro'
            monitor_mode = 'max'
            print(f"Warning: No validation data available. Monitoring {monitor_metric} instead.")
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.MODEL_SAVE_DIR,
            filename=f'{self.config.EXPERIMENT_MODE}_clf-{{epoch:02d}}-{{{monitor_metric}:.4f}}',
            save_top_k=1, 
            monitor=monitor_metric, 
            mode=monitor_mode, 
            auto_insert_metric_name=True
        )
        
        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric, 
            patience=self.config.BASE_MODEL_EARLY_STOPPING_PATIENCE, 
            mode=monitor_mode, 
            verbose=True
        )
        
        csv_logger = CSVLogger(
            save_dir=self.config.LOG_DIR, 
            name=f"{self.config.EXPERIMENT_MODE}_base_model_training"
        )

        limit_val_batches = 1.0
        if not has_validation:
            limit_val_batches = 0
            print("No validation data available. Validation will be skipped.")

        trainer = pl.Trainer(
            max_epochs=self.config.NUM_TRAIN_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=self.config.DEVICES,
            precision=self.config.PRECISION,
            logger=csv_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            deterministic=False,
            log_every_n_steps=self.config.LOG_EVERY_N_STEPS,
            gradient_clip_val=self.config.GRADIENT_CLIP_VAL,
            num_sanity_val_steps=0,
            limit_val_batches=limit_val_batches,
            check_val_every_n_epoch=1 if has_validation else None
        )

        print(f"Starting {self.config.EXPERIMENT_MODE} base model training...")
        trainer.fit(self.model, datamodule=self.data_module)
        print(f"{self.config.EXPERIMENT_MODE} base model training complete!")
        self._load_best_model(checkpoint_callback)

        del trainer
        clear_memory()

    def run_stage2_attention_extraction(self) -> Optional[pd.DataFrame]:
        if not self.config.STAGE_ATTENTION_EXTRACTION:
            print("Skipping Stage 2: Attention Extraction.")
            if self.config.STAGE_OE_EXTRACTION or self.config.STAGE_VISUALIZATION:
                try: return self._load_attention_results()
                except FileNotFoundError: print("Attention results not found.")
            return None

        print(f"\n{'='*50}\nSTAGE 2: ATTENTION EXTRACTION\n{'='*50}")
        if self.model is None: self._load_existing_model()
        if self.model is None: print("Error: Base model not available."); return None
        if self.data_module is None:
            self.data_module = EnhancedDataModule(self.config); self.data_module.setup()

        self.attention_analyzer = EnhancedAttentionAnalyzer(
            config=self.config, model_pl=self.model, tokenizer=self.data_module.tokenizer, device=get_device()
        )

        full_df = self.data_module.get_full_dataframe().copy()
        exclude_class_for_attn = self.config.EXCLUDE_CLASS_FOR_TRAINING if self.config.EXPERIMENT_MODE == "syslog" else None
        df_with_attention = self.attention_analyzer.process_full_dataset(full_df, exclude_class=exclude_class_for_attn)

        output_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_attention_{self.config.EXPERIMENT_MODE}.csv")
        df_with_attention.to_csv(output_path, index=False)
        print(f"Attention analysis results saved: {output_path}")
        self._print_attention_samples(df_with_attention)

        clear_memory()
        return df_with_attention

    def run_stage3_oe_extraction(self, df_with_attention: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
        if not self.config.STAGE_OE_EXTRACTION:
            print("Skipping Stage 3: OE & Feature Extraction.")
            if self.config.STAGE_VISUALIZATION or self.config.STAGE_OSR_EXPERIMENTS:
                try: return self._load_final_metrics_and_features()
                except FileNotFoundError: print("Final metrics/features not found.")
            return None, None

        print(f"\n{'='*50}\nSTAGE 3: OE & FEATURE EXTRACTION\n{'='*50}")
        if df_with_attention is None: df_with_attention = self._load_attention_results()
        if df_with_attention is None: print("Error: DataFrame with attention is not available."); return None, None
        if self.model is None: self._load_existing_model()
        if self.model is None: print("Error: Base model not available."); return None, None
        if self.data_module is None: self.data_module = EnhancedDataModule(self.config); self.data_module.setup()

        self.oe_extractor = OEExtractorEnhanced(
            config=self.config, model_pl=self.model, tokenizer=self.data_module.tokenizer, device=get_device()
        )

        masked_texts_col = self.config.TEXT_COLUMN_IN_OE_FILES
        if masked_texts_col not in df_with_attention.columns:
            df_with_attention[masked_texts_col] = df_with_attention[self.config.TEXT_COLUMN]

        texts_for_metrics_extraction, rows_to_skip_metric_calc = [], pd.Series([False] * len(df_with_attention), index=df_with_attention.index)
        if self.config.EXPERIMENT_MODE == "syslog":
            exclude_class_lower = self.config.EXCLUDE_CLASS_FOR_TRAINING.lower()
            rows_to_skip_metric_calc = (df_with_attention[self.config.CLASS_COLUMN].astype(str).str.lower() == exclude_class_lower)
            for idx, row in df_with_attention.iterrows():
                texts_for_metrics_extraction.append(row[self.config.TEXT_COLUMN] if rows_to_skip_metric_calc[idx] else row[masked_texts_col])
        else:
            texts_for_metrics_extraction = df_with_attention[masked_texts_col].tolist()

        max_len = self.config.MAX_LENGTH
        batch_sz = self.config.BATCH_SIZE

        metrics_dataset = MaskedTextDatasetForMetrics(texts_for_metrics_extraction, self.data_module.tokenizer, max_length=max_len, is_nlp_mode=(self.config.EXPERIMENT_MODE == "nlp"))
        metrics_dataloader = create_safe_dataloader(
            metrics_dataset, 
            batch_size=batch_sz, 
            num_workers=self.config.NUM_WORKERS, 
            shuffle=False, 
            collate_fn=self.data_module.data_collator if self.data_module.data_collator else None
        )

        attention_metrics_df_part, features_list = self.oe_extractor.extract_attention_metrics(
            metrics_dataloader, original_df_indices=df_with_attention.index, rows_to_skip_metric_calculation=rows_to_skip_metric_calc
        )

        # df_with_all_metrics = df_with_attention.copy()
        # for col in attention_metrics_df_part.columns:
        #     df_with_all_metrics[col] = attention_metrics_df_part[col].reindex(df_with_all_metrics.index)

        # if self.attention_analyzer:
        #     df_with_all_metrics = self.oe_extractor.compute_removed_word_attention(df_with_all_metrics, self.attention_analyzer, rows_to_skip_computation=rows_to_skip_metric_calc)

        df_with_all_metrics = df_with_attention.copy()
        for col in attention_metrics_df_part.columns:
            df_with_all_metrics[col] = attention_metrics_df_part[col].reindex(df_with_all_metrics.index)

        # 이 부분이 실제로 실행되는지 확인하고, self.attention_analyzer가 None이 아닌지 체크
        if self.attention_analyzer is not None:
            print("Computing removed_avg_attention scores...")  # 디버깅용 출력 추가
            df_with_all_metrics = self.oe_extractor.compute_removed_word_attention(
                df_with_all_metrics, 
                self.attention_analyzer, 
                rows_to_skip_computation=rows_to_skip_metric_calc
            )
        else:
            print("Warning: attention_analyzer is None, skipping removed_avg_attention computation")
            # 기본값으로 컬럼 추가
            df_with_all_metrics['removed_avg_attention'] = 0.0

        # 컬럼이 실제로 있는지 확인
        print(f"Columns in df_with_all_metrics: {df_with_all_metrics.columns.tolist()}")            
        
        self.oe_extractor.extract_oe_datasets(df_with_all_metrics, rows_to_exclude_from_oe=rows_to_skip_metric_calc)

        metrics_output_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_all_metrics_{self.config.EXPERIMENT_MODE}.csv")
        df_with_all_metrics.to_csv(metrics_output_path, index=False)
        print(f"DataFrame with all metrics saved: {metrics_output_path}")

        if features_list:
            features_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"extracted_features_{self.config.EXPERIMENT_MODE}.npy")
            np.save(features_path, np.array(features_list, dtype=object))
            print(f"Extracted features ({len(features_list)} samples) saved: {features_path}")

        clear_memory()
        return df_with_all_metrics, features_list

    def run_stage4_visualization(self, df_with_metrics: Optional[pd.DataFrame], features: Optional[List[np.ndarray]]):
        if not self.config.STAGE_VISUALIZATION: print("Skipping Stage 4: Visualization."); return
        print(f"\n{'='*50}\nSTAGE 4: VISUALIZATION\n{'='*50}")
        if df_with_metrics is None or features is None:
            df_with_metrics, features = self._load_final_metrics_and_features()
        if df_with_metrics is None: print("Error: DataFrame with metrics not available."); return

        self.visualizer.visualize_all_metrics(df_with_metrics)
        if features and self.data_module:
            if self.data_module.label2id is None: self.data_module.setup()
            if self.data_module.label2id:
                 self.visualizer.visualize_oe_candidates(df_with_metrics, features, self.data_module.label2id, self.data_module.id2label)

        print("Visualization of attention-derived OE metrics complete!")
        clear_memory()

    def run_stage5_osr_experiments(self):
        if not self.config.STAGE_OSR_EXPERIMENTS: 
            print("Skipping Stage 5: OSR Experiments."); 
            return

        print(f"\n{'='*50}\nSTAGE 5: OSR EXPERIMENTS ({self.config.EXPERIMENT_MODE.upper()})\n{'='*50}")

        if self.config.EXPERIMENT_MODE == "nlp":
            if self.config.COMPREHENSIVE_NLP_EXPERIMENTS:
                self._run_comprehensive_nlp_osr_experiments()
            else:
                self._run_single_nlp_osr_experiment()
        else:
            self._run_syslog_osr_experiments()

        clear_memory()

    def _run_comprehensive_nlp_osr_experiments(self):
        """모든 NLP 데이터셋을 각각 ID로 사용하여 OSR 실험을 수행합니다."""
        print("\n=== 🚀 COMPREHENSIVE NLP OSR EXPERIMENTS 🚀 ===")
        print("Running OSR experiments with each NLP dataset as ID dataset...")
        
        datasets_to_experiment = self.comprehensive_nlp_manager.get_datasets_for_comprehensive_experiments()
        print(f"Datasets for comprehensive experiments: {datasets_to_experiment}")
        
        all_comprehensive_results = {}
        
        try:
            for dataset_idx, id_dataset_name in enumerate(datasets_to_experiment):
                print(f"\n{'='*70}")
                print(f"🔄 EXPERIMENT {dataset_idx + 1}/{len(datasets_to_experiment)}: ID Dataset = {id_dataset_name}")
                print(f"{'='*70}")
                
                # Config 설정 업데이트
                self.comprehensive_nlp_manager.setup_config_for_dataset(id_dataset_name)
                print(f"✅ Config updated for dataset: {id_dataset_name}")
                
                # 해당 데이터셋으로 전체 파이프라인 실행
                try:
                    # 상태 초기화
                    self._reset_pipeline_state()
                    
                    print(f"🔧 Running full pipeline for {id_dataset_name}...")
                    
                    # Stage 1: Model Training
                    self.run_stage1_model_training()
                    self._cleanup_after_experiment()
                    
                    # Stage 2: Attention Extraction
                    df_with_attention = self.run_stage2_attention_extraction()
                    self._cleanup_after_experiment()
                    
                    # Stage 3: OE Extraction
                    df_with_metrics, features = self.run_stage3_oe_extraction(df_with_attention)
                    self._cleanup_after_experiment()
                    
                    # Stage 4: Visualization
                    self.run_stage4_visualization(df_with_metrics, features)
                    
                    # Stage 5: OSR Experiments (단일 데이터셋)
                    print(f"🎯 Running OSR experiments for {id_dataset_name}...")
                    dataset_results = self._run_single_nlp_osr_experiment()
                    
                    # 결과 저장
                    if dataset_results:
                        for key, value in dataset_results.items():
                            comprehensive_key = f"{id_dataset_name}|{key.split('|', 1)[1]}"
                            all_comprehensive_results[comprehensive_key] = value
                    
                    print(f"✅ Completed experiments for {id_dataset_name}")
                    
                except Exception as e:
                    print(f"❌ Error during experiment with {id_dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                finally:
                    self._cleanup_after_experiment()
        
        finally:
            # Config 복원
            self.comprehensive_nlp_manager.restore_original_config()
            print("📋 Configuration restored to original state.")
        
        # 종합 결과 저장
        if all_comprehensive_results:
            print(f"\n🎉 COMPREHENSIVE NLP OSR EXPERIMENTS COMPLETED!")
            print(f"Total experiments: {len(datasets_to_experiment)} ID datasets")
            print(f"Total result combinations: {len(all_comprehensive_results)}")
            
            self._save_comprehensive_osr_results(all_comprehensive_results, "comprehensive_nlp")
        else:
            print("❌ No comprehensive results to save.")

    def _run_single_nlp_osr_experiment(self) -> Dict:
        """단일 NLP 데이터셋에 대한 OSR 실험을 실행하고 결과를 반환합니다."""
        if self.data_module is None or self.data_module.num_labels is None:
            self.data_module = EnhancedDataModule(self.config)
            self.data_module.prepare_data()
            self.data_module.setup()
            if self.data_module.num_labels is None: 
                print("Critical Error: Failed to set up DataModule.")
                return {}

        return self._run_nlp_osr_experiments()

    def _count_tokens_for_fair_comparison(self, texts: List[str], tokenizer) -> int:
        """실제 토크나이저를 사용하여 토큰 수를 계산 - 효율적인 배치 처리"""
        if not texts:
            return 0
        
        batch_size = 1000
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                # truncation=True 추가하여 최대 길이로 자르기
                tokenized = tokenizer.batch_encode_plus(
                    batch, 
                    add_special_tokens=True, 
                    truncation=True,  # False → True로 변경
                    max_length=self.config.OSR_MAX_LENGTH,  # 최대 길이 지정
                    padding=False,
                    return_attention_mask=False
                )
                total_tokens += sum(len(ids) for ids in tokenized['input_ids'])
            except Exception as e:
                print(f"Warning: Error in batch tokenization: {e}")
                # 폴백: 개별 처리
                for text in batch:
                    try:
                        tokens = tokenizer.encode(
                            text, 
                            add_special_tokens=True,
                            truncation=True,  # 여기도 추가
                            max_length=self.config.OSR_MAX_LENGTH
                        )
                        total_tokens += len(tokens)
                    except:
                        # 최악의 경우 추정치 사용
                        total_tokens += min(self.config.OSR_MAX_LENGTH, int(len(text.split()) * 1.5))
                            
        return total_tokens

    def _train_single_osr_model_lightning(self, oe_tag, num_classes, tokenizer, id_train_loader, id_val_loader, oe_dataset, model_save_path):
        """PyTorch Lightning을 사용한 OSR 모델 학습"""
        print(f"\n===== Training OSR Model for OE Source: {oe_tag} =====")

        if os.path.exists(model_save_path) and not self.config.OSR_FORCE_RETRAIN:
            print(f"Model already exists at {model_save_path}. Skipping training.")
            return

        # Lightning Module 생성
        osr_module = OSRLightningModule(
            model_name=self.config.OSR_MODEL_TYPE,
            num_labels=num_classes,
            learning_rate=self.config.OSR_LEARNING_RATE,
            oe_lambda=self.config.OSR_OE_LAMBDA,
            temperature=self.config.OSR_TEMPERATURE,
            cache_dir=self.config.HUGGINGFACE_CACHE_DIR
        )

        # OE DataLoader 준비
        oe_loader = None
        if oe_dataset:
            oe_loader = create_safe_dataloader(
                oe_dataset, 
                batch_size=self.config.OSR_BATCH_SIZE, 
                shuffle=True, 
                num_workers=self.config.OSR_NUM_DATALOADER_WORKERS
            )

        # Combined DataLoader for training
        train_loader = CombinedOSRDataLoader(id_train_loader, oe_loader)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.dirname(model_save_path),
            filename=os.path.splitext(os.path.basename(model_save_path))[0],
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            save_last=True
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.OSR_EARLY_STOPPING_PATIENCE,
            mode='min',
            min_delta=self.config.OSR_EARLY_STOPPING_MIN_DELTA
        )

        # Logger
        csv_logger = CSVLogger(
            save_dir=self.config.LOG_DIR,
            name=f"osr_training_{oe_tag}"
        )

        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.config.OSR_NUM_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=self.config.DEVICES,
            precision=self.config.PRECISION,
            callbacks=[checkpoint_callback, early_stopping],
            logger=csv_logger,
            gradient_clip_val=self.config.GRADIENT_CLIP_VAL,
            log_every_n_steps=10,
            enable_progress_bar=True,
            deterministic=False
        )

        # Training
        print(f"Training for up to {self.config.OSR_NUM_EPOCHS} epochs with early stopping...")
        trainer.fit(osr_module, train_dataloaders=train_loader, val_dataloaders=id_val_loader)

        # Save best model state dict
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix from keys
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            torch.save(state_dict, model_save_path)
            print(f"Best model saved to {model_save_path}")

        # Cleanup
        del osr_module, trainer
        clear_memory()

    def _prepare_final_oe_scenarios_for_fair_comparison(self, base_oe_scenarios, tokenizer):
        """
        Handles the logic for Fair Comparison mode (token budget sampling).
        This function is now generalized to be used by both NLP and Syslog OSR experiments.
        """
        final_oe_scenarios = []
        if self.config.OSR_FAIR_COMPARISON_MODE:
            print("\n--- FAIR COMPARISON MODE (TOKEN BUDGET) ENABLED ---")

            standard_model = next((s for s in base_oe_scenarios if s['tag'] == 'Standard'), None)
            if standard_model:
                final_oe_scenarios.append(standard_model)

            non_standard_scenarios = [s for s in base_oe_scenarios if s['tag'] != 'Standard' and s.get('dataset') is not None and s['count'] > 0]

            if not non_standard_scenarios:
                print("Warning: No valid OE scenarios found for fair comparison. Running standard mode.")
                return base_oe_scenarios
            else:
                print("Calculating token budgets using the OSR model's tokenizer...")
                for scenario in tqdm(non_standard_scenarios, desc="Calculating token budgets"):
                    if isinstance(scenario['dataset'], Subset):
                        texts = [scenario['dataset'].dataset.texts[i] for i in scenario['dataset'].indices]
                    else:
                        texts = scenario['dataset'].texts
                    scenario['total_tokens'] = self._count_tokens_for_fair_comparison(texts, tokenizer)

                min_token_budget = min(s['total_tokens'] for s in non_standard_scenarios if s['total_tokens'] > 0)
                print(f"\nAll OE sets will be sampled to match the minimum token budget: {min_token_budget}")

                for scenario in non_standard_scenarios:
                    original_dataset = scenario['dataset']
                    if scenario['total_tokens'] > min_token_budget:
                        print(f"  - Sampling '{scenario['tag']}' down to token budget...")

                        original_texts_with_indices = []
                        if isinstance(original_dataset, Subset):
                             original_indices_subset = original_dataset.indices
                             underlying_dataset = original_dataset.dataset
                             original_texts_with_indices = [(idx, underlying_dataset.texts[idx]) for idx in original_indices_subset]
                        else:
                             underlying_dataset = original_dataset
                             original_texts_with_indices = list(enumerate(underlying_dataset.texts))

                        random.shuffle(original_texts_with_indices)

                        sampled_texts_for_count = []
                        sampled_original_indices = []

                        for original_idx, text in original_texts_with_indices:
                            sampled_texts_for_count.append(text)
                            sampled_original_indices.append(original_idx)

                            if len(sampled_texts_for_count) % 500 == 0 or len(sampled_texts_for_count) == len(original_texts_with_indices):
                                current_tokens = self._count_tokens_for_fair_comparison(sampled_texts_for_count, tokenizer)
                                if current_tokens >= min_token_budget:
                                    break

                        final_sampled_dataset = Subset(underlying_dataset, sampled_original_indices)
                        new_tag = f"{scenario['tag']}_TokenBudget"
                        final_oe_scenarios.append({'tag': new_tag, 'dataset': final_sampled_dataset, 'count': len(final_sampled_dataset)})
                    else:
                        print(f"  - Using '{scenario['tag']}' as is, since its token count ({scenario['total_tokens']}) is at or below the budget.")
                        final_oe_scenarios.append(scenario)
        else:
            print("\n--- Standard Experiment Mode ---")
            final_oe_scenarios = base_oe_scenarios

        print("\n--- Analyzing Final OE Dataset Statistics ---")
        for scenario in final_oe_scenarios:
            if scenario.get('dataset') is not None:
                texts = []
                if isinstance(scenario['dataset'], Subset):
                    original_dataset = scenario['dataset'].dataset
                    indices = scenario['dataset'].indices
                    texts = [original_dataset.texts[i] for i in indices]
                else:
                    texts = scenario['dataset'].texts

                if not texts:
                    print(f"  - Scenario: {scenario['tag']} - No texts found.")
                    continue

                total_tokens = self._count_tokens_for_fair_comparison(texts, tokenizer)
                unique_word_tokens = set()
                for text in texts:
                    unique_word_tokens.update(text.lower().split())
                avg_tokens_per_sample = total_tokens / len(texts) if texts else 0

                print(f"  - Scenario: {scenario['tag']}")
                print(f"    - Sample Count: {scenario['count']}")
                print(f"    - Total Tokens (by model tokenizer): {total_tokens}")
                print(f"    - Unique Word Tokens (approx.): {len(unique_word_tokens)}")
                print(f"    - Avg. Tokens/Sample: {avg_tokens_per_sample:.2f}")

        return final_oe_scenarios

    def _run_nlp_osr_experiments(self) -> Dict:
        """NLP OSR 실험 실행"""
        print("\n--- Running NLP OSR Experiments (Train Once, Evaluate Many) ---")
        osr_nlp_tokenizer = RobertaTokenizer.from_pretrained(self.config.OSR_MODEL_TYPE)

        # 데이터 준비
        id_train_ds, id_test_ds, num_id_classes, id_l2i, id_i2l = \
            prepare_nlp_id_data_for_osr(self.data_module, osr_nlp_tokenizer, self.config.OSR_MAX_LENGTH)
        if id_train_ds is None or num_id_classes == 0:
            print("Error: Failed to prepare NLP ID data."); return {}

        id_train_loader = create_safe_dataloader(id_train_ds, batch_size=self.config.OSR_BATCH_SIZE, shuffle=True, num_workers=self.config.OSR_NUM_DATALOADER_WORKERS)
        id_test_loader = create_safe_dataloader(id_test_ds, batch_size=self.config.OSR_BATCH_SIZE, shuffle=False, num_workers=self.config.OSR_NUM_DATALOADER_WORKERS)
        id_test_count = len(id_test_ds)

        # 모든 OOD 데이터셋 로드
        all_ood_datasets = {}
        ood_test_datasets = ['wmt16', 'wikitext', 'trec', 'sst2', 'syslog']  # syslog 추가
        for ood_name in ood_test_datasets:
            if ood_name == 'syslog':
                # Syslog unknown as OOD
                ood_dataset = prepare_syslog_as_ood_for_nlp(osr_nlp_tokenizer, self.config.OSR_MAX_LENGTH)
                if ood_dataset:
                    all_ood_datasets['Syslog_Unknown'] = {
                        "dataset": ood_dataset,
                        "count": len(ood_dataset)
                    }
            else:
                ood_data = NLPDatasetLoader.load_any_dataset(ood_name, split='test')
                if ood_data and ood_data['text']:
                    ood_ds = OSRNLPDataset(ood_data['text'], [-1] * len(ood_data['text']), osr_nlp_tokenizer, self.config.OSR_MAX_LENGTH)
                    all_ood_datasets[ood_name] = {
                        "dataset": ood_ds,
                        "count": len(ood_ds)
                    }
        
        if not all_ood_datasets:
            print("Warning: No OOD evaluation data available.")
            return {}

        # OOD 균형화 적용
        original_sizes = {name: data['count'] for name, data in all_ood_datasets.items()}
        target_sizes = compute_balance_target_sizes(all_ood_datasets, self.config.OOD_SAMPLING_STRATEGY, self.config, id_test_count)
        
        all_ood_loaders = {}
        balanced_sizes = {}
        
        for ood_name, ood_data in all_ood_datasets.items():
            target_size = target_sizes[ood_name]
            balanced_dataset = balance_ood_dataset(
                ood_data["dataset"], 
                target_size, 
                ood_name, 
                self.config.RANDOM_STATE
            )
            
            all_ood_loaders[ood_name] = {
                "loader": create_safe_dataloader(
                    balanced_dataset, 
                    batch_size=self.config.OSR_BATCH_SIZE, 
                    shuffle=False, 
                    num_workers=self.config.OSR_NUM_DATALOADER_WORKERS
                ),
                "count": len(balanced_dataset),
                "original_count": ood_data["count"]
            }
            balanced_sizes[ood_name] = len(balanced_dataset)
        
        # 균형화 통계 출력
        if self.config.REPORT_OOD_BALANCE_STATS:
            print_ood_balance_stats(original_sizes, target_sizes, balanced_sizes)

        # OE 시나리오 정의
        base_oe_scenarios = []
        if not self.config.OSR_SKIP_STANDARD_MODEL:
            base_oe_scenarios.append({'tag': 'Standard', 'dataset': None, 'count': 0})

        for oe_name in self.config.OSR_EXTERNAL_OE_DATASETS:
            oe_data = NLPDatasetLoader.load_any_dataset(oe_name, split='train')
            if oe_data and oe_data['text']:
                oe_ds = OSRNLPDataset(oe_data['text'], [-1] * len(oe_data['text']), osr_nlp_tokenizer, self.config.OSR_MAX_LENGTH)
                base_oe_scenarios.append({'tag': f"OE_{oe_name}", 'dataset': oe_ds, 'count': len(oe_ds)})

        attn_oe_files = [f for f in os.listdir(self.config.OE_DATA_DIR) if f.endswith('.csv') and 'extended' not in f]
        for oe_file in attn_oe_files:
            oe_tag = os.path.splitext(oe_file)[0].replace("oe_data_", "")
            oe_data_path = os.path.join(self.config.OE_DATA_DIR, oe_file)
            attn_oe_ds = prepare_attention_derived_oe_data_for_osr(osr_nlp_tokenizer, self.config.OSR_MAX_LENGTH, oe_data_path, self.config.TEXT_COLUMN_IN_OE_FILES, for_syslog=False)
            if attn_oe_ds:
                base_oe_scenarios.append({'tag': f"Attn_{oe_tag}", 'dataset': attn_oe_ds, 'count': len(attn_oe_ds)})

        final_oe_scenarios = self._prepare_final_oe_scenarios_for_fair_comparison(base_oe_scenarios, osr_nlp_tokenizer)

        # 모델 학습 단계
        if not self.config.OSR_EVAL_ONLY:
            print(f"\n--- STAGE 5.1: OSR Model Training Phase ---")
            for scenario in final_oe_scenarios:
                oe_tag = scenario['tag']
                sanitized_oe_tag = re.sub(r'[^\w\-.()]+', '_', oe_tag)
                model_save_path = os.path.join(self.config.OSR_MODEL_DIR, f"osr_model_NLP_{sanitized_oe_tag}.pt")

                self._train_single_osr_model_lightning(
                    oe_tag=oe_tag,
                    num_classes=num_id_classes,
                    tokenizer=osr_nlp_tokenizer,
                    id_train_loader=id_train_loader,
                    id_val_loader=id_test_loader,
                    oe_dataset=scenario.get('dataset'),
                    model_save_path=model_save_path
                )
        else:
            print("\n--- STAGE 5.1: OSR Model Training SKIPPED (OSR_EVAL_ONLY=True) ---")

        # 모델 평가 단계
        print(f"\n--- STAGE 5.2: OSR Model Evaluation Phase ---")
        print(f"OOD Balancing Strategy: {self.config.OOD_SAMPLING_STRATEGY}")
        
        all_nlp_osr_results = {}

        device = get_osr_device()
        for scenario in final_oe_scenarios:
            oe_tag = scenario['tag']
            oe_count = scenario['count']
            sanitized_oe_tag = re.sub(r'[^\w\-.()]+', '_', oe_tag)
            model_save_path = os.path.join(self.config.OSR_MODEL_DIR, f"osr_model_NLP_{sanitized_oe_tag}.pt")

            if not os.path.exists(model_save_path):
                print(f"Warning: Model for '{oe_tag}' not found at {model_save_path}. Skipping evaluation.")
                continue

            print(f"\n===== Evaluating Model from OE Source: {oe_tag} =====")
            model_osr = RoBERTaOOD(
                model_name=self.config.OSR_MODEL_TYPE,
                num_labels=num_id_classes,
                cache_dir=self.config.HUGGINGFACE_CACHE_DIR
            )
            model_osr.load_state_dict(torch.load(model_save_path, map_location=device))
            model_osr.to(device)

            for ood_name, ood_data in all_ood_loaders.items():
                ood_loader = ood_data["loader"]
                ood_count = ood_data["count"]
                original_ood_count = ood_data["original_count"]
                
                print(f"  -> Evaluating against OOD: {ood_name} ({ood_count} samples, original: {original_ood_count})")

                exp_subdir = os.path.join("NLP", f"OE_{sanitized_oe_tag}_vs_OOD_{ood_name}")
                current_run_result_dir = os.path.join(self.config.OSR_RESULT_DIR, exp_subdir)
                ensure_directory(current_run_result_dir)

                eval_results, eval_plot_data = evaluate_osr(
                    model_osr, id_test_loader, ood_loader, device,
                    self.config.OSR_TEMPERATURE, self.config.OSR_THRESHOLD_PERCENTILE, return_data=True, mode="nlp"
                )

                # 결과에 균형화 정보 추가
                eval_results['ID_Count'] = id_test_count
                eval_results['OE_Count'] = oe_count
                eval_results['OOD_Count'] = ood_count
                eval_results['OOD_Original_Count'] = original_ood_count
                eval_results['OOD_Balance_Strategy'] = self.config.OOD_SAMPLING_STRATEGY

                metric_key = f"NLP|{oe_tag}|{ood_name}"
                all_nlp_osr_results[metric_key] = eval_results

                if not self.config.OSR_NO_PLOT_PER_EXPERIMENT:
                    plot_prefix = f"NLP_{sanitized_oe_tag}_vs_{ood_name}"
                    plot_title_suffix = f"NLP | OE: {oe_tag} | OOD: {ood_name}"
                    
                    # 균형화 정보를 플롯 제목에 추가
                    if self.config.OOD_SAMPLING_STRATEGY != 'original':
                        plot_title_suffix += f" | Balanced: {self.config.OOD_SAMPLING_STRATEGY}"
                    
                    if eval_plot_data.get("id_scores", np.array([])).size > 0 and eval_plot_data.get("ood_scores", np.array([])).size > 0:
                        plot_confidence_histograms_osr(
                            eval_plot_data["id_scores"], 
                            eval_plot_data["ood_scores"], 
                            f'Confidence - {plot_title_suffix}', 
                            os.path.join(current_run_result_dir, f'{plot_prefix}_conf_hist.png')
                        )
                        plot_roc_curve_osr(
                            eval_plot_data["id_scores"], 
                            eval_plot_data["ood_scores"], 
                            f'ROC Curve - {plot_title_suffix}', 
                            os.path.join(current_run_result_dir, f'{plot_prefix}_roc.png')
                        )

            del model_osr
            clear_memory()

        # 결과 저장
        experiment_name = f"nlp_{self.config.CURRENT_NLP_DATASET}_balanced_{self.config.OOD_SAMPLING_STRATEGY}"
        if self.config.OOD_SAMPLING_STRATEGY == 'fixed':
            experiment_name += f"_{self.config.OOD_SAMPLE_SIZE}"
        elif self.config.OOD_SAMPLING_STRATEGY == 'proportional':
            experiment_name += f"_{self.config.OOD_PROPORTIONAL_RATIO}"
        
        self._save_osr_results(all_nlp_osr_results, experiment_name)
        return all_nlp_osr_results

    def _run_syslog_osr_experiments(self):
        print("\n--- Running Syslog OSR Experiments (Train Once, Evaluate Many) ---")
        osr_syslog_tokenizer = RobertaTokenizer.from_pretrained(self.config.OSR_MODEL_TYPE)

        # 데이터 준비
        id_train_ds, id_test_ds, num_id_classes, id_l2i, id_i2l = \
            prepare_syslog_id_data_for_osr(self.data_module, osr_syslog_tokenizer, self.config.OSR_MAX_LENGTH)
        if id_train_ds is None or num_id_classes == 0:
            print("Error: Failed to prepare Syslog ID data."); return

        id_train_loader = create_safe_dataloader(id_train_ds, batch_size=self.config.OSR_BATCH_SIZE, shuffle=True, num_workers=self.config.OSR_NUM_DATALOADER_WORKERS)
        id_test_loader = create_safe_dataloader(id_test_ds, batch_size=self.config.OSR_BATCH_SIZE, shuffle=False, num_workers=self.config.OSR_NUM_DATALOADER_WORKERS)
        id_test_count = len(id_test_ds)

        # 모든 OOD 데이터셋 로드
        all_ood_datasets = {}
        
        # 1. Syslog unknown 클래스 추가
        syslog_unknown_dataset = prepare_syslog_unknown_as_ood(osr_syslog_tokenizer, self.config.OSR_MAX_LENGTH)
        if syslog_unknown_dataset:
            all_ood_datasets['Syslog_Unknown'] = {
                "dataset": syslog_unknown_dataset,
                "count": len(syslog_unknown_dataset)
            }
        
        # 2. 외부 NLP 데이터셋들
        ood_test_datasets = ['wmt16', 'wikitext', '20newsgroups', 'trec', 'sst2']
        for ood_name in ood_test_datasets:
            ood_dataset = prepare_nlp_as_ood_for_syslog(ood_name, osr_syslog_tokenizer, self.config.OSR_MAX_LENGTH)
            if ood_dataset:
                all_ood_datasets[ood_name] = {
                    "dataset": ood_dataset,
                    "count": len(ood_dataset)
                }
        
        if not all_ood_datasets:
            print("Warning: No OOD evaluation data available.")
            return

        # OOD 균형화 적용
        original_sizes = {name: data['count'] for name, data in all_ood_datasets.items()}
        target_sizes = compute_balance_target_sizes(all_ood_datasets, self.config.OOD_SAMPLING_STRATEGY, self.config, id_test_count)
        
        all_ood_loaders = {}
        balanced_sizes = {}
        
        for ood_name, ood_data in all_ood_datasets.items():
            target_size = target_sizes[ood_name]
            balanced_dataset = balance_ood_dataset(
                ood_data["dataset"], 
                target_size, 
                ood_name, 
                self.config.RANDOM_STATE
            )
            
            all_ood_loaders[ood_name] = {
                "loader": create_safe_dataloader(
                    balanced_dataset, 
                    batch_size=self.config.OSR_BATCH_SIZE, 
                    shuffle=False, 
                    num_workers=self.config.OSR_NUM_DATALOADER_WORKERS
                ),
                "count": len(balanced_dataset),
                "original_count": ood_data["count"]
            }
            balanced_sizes[ood_name] = len(balanced_dataset)
        
        # 균형화 통계 출력
        if self.config.REPORT_OOD_BALANCE_STATS:
            print_ood_balance_stats(original_sizes, target_sizes, balanced_sizes)

        # OE 시나리오 정의
        base_oe_scenarios = []
        if not self.config.OSR_SKIP_STANDARD_MODEL:
            base_oe_scenarios.append({'tag': 'Standard', 'dataset': None, 'count': 0})

        for oe_name in self.config.OSR_EXTERNAL_OE_DATASETS:
            oe_data = NLPDatasetLoader.load_any_dataset(oe_name, split='train')
            if oe_data and oe_data['text']:
                oe_ds = OSRSyslogTextDataset(oe_data['text'], [-1] * len(oe_data['text']), osr_syslog_tokenizer, self.config.OSR_MAX_LENGTH)
                base_oe_scenarios.append({'tag': f"OE_{oe_name}", 'dataset': oe_ds, 'count': len(oe_ds)})

        attn_oe_files = [f for f in os.listdir(self.config.OE_DATA_DIR) if f.endswith('.csv') and 'extended' not in f]
        for oe_file in attn_oe_files:
            oe_tag = os.path.splitext(oe_file)[0].replace("oe_data_", "")
            oe_data_path = os.path.join(self.config.OE_DATA_DIR, oe_file)
            attn_oe_ds = prepare_attention_derived_oe_data_for_osr(osr_syslog_tokenizer, self.config.OSR_MAX_LENGTH, oe_data_path, self.config.TEXT_COLUMN_IN_OE_FILES, for_syslog=True)
            if attn_oe_ds:
                base_oe_scenarios.append({'tag': f"Attn_{oe_tag}", 'dataset': attn_oe_ds, 'count': len(attn_oe_ds)})

        final_oe_scenarios = self._prepare_final_oe_scenarios_for_fair_comparison(base_oe_scenarios, osr_syslog_tokenizer)

        # 모델 학습 단계
        if not self.config.OSR_EVAL_ONLY:
            print(f"\n--- STAGE 5.1: OSR Model Training Phase ---")
            for scenario in final_oe_scenarios:
                oe_tag = scenario['tag']
                sanitized_oe_tag = re.sub(r'[^\w\-.()]+', '_', oe_tag)
                model_save_path = os.path.join(self.config.OSR_MODEL_DIR, f"osr_model_Syslog_{sanitized_oe_tag}.pt")

                self._train_single_osr_model_lightning(
                    oe_tag=oe_tag,
                    num_classes=num_id_classes,
                    tokenizer=osr_syslog_tokenizer,
                    id_train_loader=id_train_loader,
                    id_val_loader=id_test_loader,
                    oe_dataset=scenario.get('dataset'),
                    model_save_path=model_save_path
                )
        else:
            print("\n--- STAGE 5.1: OSR Model Training SKIPPED (OSR_EVAL_ONLY=True) ---")

        # 모델 평가 단계
        print(f"\n--- STAGE 5.2: OSR Model Evaluation Phase ---")
        print(f"OOD Balancing Strategy: {self.config.OOD_SAMPLING_STRATEGY}")
        
        all_syslog_osr_results = {}

        device = get_osr_device()
        for scenario in final_oe_scenarios:
            oe_tag = scenario['tag']
            oe_count = scenario['count']
            sanitized_oe_tag = re.sub(r'[^\w\-.()]+', '_', oe_tag)
            model_save_path = os.path.join(self.config.OSR_MODEL_DIR, f"osr_model_Syslog_{sanitized_oe_tag}.pt")

            if not os.path.exists(model_save_path):
                print(f"Warning: Model for '{oe_tag}' not found at {model_save_path}. Skipping evaluation.")
                continue

            print(f"\n===== Evaluating Model from OE Source: {oe_tag} =====")
            model_osr = RoBERTaOOD(
                model_name=self.config.OSR_MODEL_TYPE,
                num_labels=num_id_classes,
                cache_dir=self.config.HUGGINGFACE_CACHE_DIR
            )
            model_osr.load_state_dict(torch.load(model_save_path, map_location=device))
            model_osr.to(device)

            for ood_name, ood_data in all_ood_loaders.items():
                ood_loader = ood_
                ood_loader = ood_data["loader"]
                ood_count = ood_data["count"]
                original_ood_count = ood_data["original_count"]
                
                print(f"  -> Evaluating against OOD: {ood_name} ({ood_count} samples, original: {original_ood_count})")

                exp_subdir = os.path.join("SYSLOG", f"OE_{sanitized_oe_tag}_vs_OOD_{ood_name}")
                current_run_result_dir = os.path.join(self.config.OSR_RESULT_DIR, exp_subdir)
                ensure_directory(current_run_result_dir)

                eval_results, eval_plot_data = evaluate_osr(
                    model_osr, id_test_loader, ood_loader, device,
                    self.config.OSR_TEMPERATURE, self.config.OSR_THRESHOLD_PERCENTILE, return_data=True, mode="syslog"
                )

                # 결과에 균형화 정보 추가
                eval_results['ID_Count'] = id_test_count
                eval_results['OE_Count'] = oe_count
                eval_results['OOD_Count'] = ood_count
                eval_results['OOD_Original_Count'] = original_ood_count
                eval_results['OOD_Balance_Strategy'] = self.config.OOD_SAMPLING_STRATEGY

                metric_key = f"Syslog|{oe_tag}|{ood_name}"
                all_syslog_osr_results[metric_key] = eval_results

                if not self.config.OSR_NO_PLOT_PER_EXPERIMENT:
                    plot_prefix = f"Syslog_{sanitized_oe_tag}_vs_{ood_name}"
                    plot_title_suffix = f"Syslog | OE: {oe_tag} | OOD: {ood_name}"
                    
                    # 균형화 정보를 플롯 제목에 추가
                    if self.config.OOD_SAMPLING_STRATEGY != 'original':
                        plot_title_suffix += f" | Balanced: {self.config.OOD_SAMPLING_STRATEGY}"
                    
                    if eval_plot_data.get("id_scores", np.array([])).size > 0 and eval_plot_data.get("ood_scores", np.array([])).size > 0:
                        plot_confidence_histograms_osr(
                            eval_plot_data["id_scores"], 
                            eval_plot_data["ood_scores"], 
                            f'Confidence - {plot_title_suffix}', 
                            os.path.join(current_run_result_dir, f'{plot_prefix}_conf_hist.png')
                        )
                        plot_roc_curve_osr(
                            eval_plot_data["id_scores"], 
                            eval_plot_data["ood_scores"], 
                            f'ROC Curve - {plot_title_suffix}', 
                            os.path.join(current_run_result_dir, f'{plot_prefix}_roc.png')
                        )

            del model_osr
            clear_memory()

        # 결과 저장
        experiment_name = f"syslog_balanced_{self.config.OOD_SAMPLING_STRATEGY}"
        if self.config.OOD_SAMPLING_STRATEGY == 'fixed':
            experiment_name += f"_{self.config.OOD_SAMPLE_SIZE}"
        elif self.config.OOD_SAMPLING_STRATEGY == 'proportional':
            experiment_name += f"_{self.config.OOD_PROPORTIONAL_RATIO}"
        
        self._save_osr_results(all_syslog_osr_results, experiment_name)

    def _save_comprehensive_osr_results(self, results_dict: Dict, experiment_group_name: str):
        """종합 NLP 실험 결과를 저장하는 메소드"""
        if not results_dict: 
            print("No comprehensive OSR metrics generated to save.")
            return
            
        print(f"\n===== COMPREHENSIVE OSR Summary ({experiment_group_name}) =====")
        
        records = []
        for key, metrics in results_dict.items():
            try:
                parts = key.split('|')
                if len(parts) >= 3:
                    id_part, oe_part, ood_part = parts[0], parts[1], parts[2]
                    record = {
                        'ID_Dataset': id_part,
                        'OE_Source': oe_part,
                        'OOD_Dataset': ood_part,
                        **metrics
                    }
                    records.append(record)
                else:
                    print(f"Warning: Could not parse result key '{key}'. Adding as raw row.")
                    records.append({'Name': key, **metrics})
            except ValueError:
                print(f"Warning: Could not parse result key '{key}'. Adding as raw row.")
                records.append({'Name': key, **metrics})

        results_df = pd.DataFrame(records)
        
        # 컬럼 순서 정리
        standard_cols_order = [
            'ID_Count', 'OE_Count', 'OOD_Count', 'Closed_Set_Accuracy', 'F1_Macro',
            'AUROC_OOD', 'FPR@TPR95', 'AUPR_In', 'AUPR_Out', 'OSCR', 'DetectionAccuracy', 'Threshold_Used'
        ]
        
        if 'Name' in results_df.columns:
            base_cols = ['Name']
        else:
            base_cols = ['ID_Dataset', 'OE_Source', 'OOD_Dataset']
        
        metric_cols = [col for col in results_df.columns if col not in base_cols]
        ordered_metric_cols = [col for col in standard_cols_order if col in metric_cols] + [col for col in metric_cols if col not in standard_cols_order]
        final_ordered_cols = base_cols + ordered_metric_cols
        
        results_df = results_df[[col for col in final_ordered_cols if col in results_df.columns]]

        print("Comprehensive OSR Performance Metrics DataFrame:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(results_df)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fair_mode_suffix = "_FAIR" if self.config.OSR_FAIR_COMPARISON_MODE else ""
        base_filename = f"comprehensive_osr_summary_{experiment_group_name}{fair_mode_suffix}_{timestamp}"
        
        # 메인 결과 디렉토리에 저장 (각 데이터셋 디렉토리가 아닌)
        main_result_dir = os.path.join(self.config.BASE_OUTPUT_DIR, "comprehensive_results")
        ensure_directory(main_result_dir)
        
        csv_path = os.path.join(main_result_dir, f"{base_filename}.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.8f')
        print(f"\n🎉 Comprehensive OSR results saved to CSV: {csv_path}")
        
        # Pivot tables 생성
        self._save_pivot_tables(results_df, main_result_dir, base_filename)
        
        # 설정과 결과를 JSON으로도 저장
        osr_config_subset = { k: getattr(self.config, k) for k in dir(self.config) if k.startswith('OSR_') or k in ['RANDOM_STATE', 'EXPERIMENT_MODE', 'COMPREHENSIVE_NLP_EXPERIMENTS', 'NLP_DATASETS_FOR_COMPREHENSIVE']}
        with open(os.path.join(main_result_dir, f"{base_filename}_config.json"), 'w') as f:
            json.dump({'osr_config': osr_config_subset, 'results_dataframe': results_df.to_dict(orient='records')}, f, indent=2, default=str)
        print(f"📋 Comprehensive OSR results and config saved to JSON.")

    def _save_osr_results(self, results_dict: Dict, experiment_group_name: str):
        if not results_dict: print("No OSR metrics generated to save."); return
        print(f"\n===== OSR Summary ({experiment_group_name}) =====")

        records = []
        for key, metrics in results_dict.items():
            try:
                id_part, oe_part, ood_part = key.split('|')
                record = {
                    'ID_Dataset': id_part,
                    'OE_Source': oe_part,
                    'OOD_Dataset': ood_part,
                    **metrics
                }
                records.append(record)
            except ValueError:
                print(f"Warning: Could not parse result key '{key}'. Adding as raw row.")
                records.append({'Name': key, **metrics})

        results_df = pd.DataFrame(records)

        standard_cols_order = [
            'ID_Count', 'OE_Count', 'OOD_Count', 'Closed_Set_Accuracy', 'F1_Macro',
            'AUROC_OOD', 'FPR@TPR95', 'AUPR_In', 'AUPR_Out', 'OSCR', 'DetectionAccuracy', 'Threshold_Used'
        ]

        if 'Name' in results_df.columns:
            base_cols = ['Name']
        else:
            base_cols = ['ID_Dataset', 'OE_Source', 'OOD_Dataset']

        metric_cols = [col for col in results_df.columns if col not in base_cols]
        ordered_metric_cols = [col for col in standard_cols_order if col in metric_cols] + [col for col in metric_cols if col not in standard_cols_order]
        final_ordered_cols = base_cols + ordered_metric_cols

        results_df = results_df[[col for col in final_ordered_cols if col in results_df.columns]]

        print("Overall OSR Performance Metrics DataFrame:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(results_df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        fair_mode_suffix = "_FAIR" if self.config.OSR_FAIR_COMPARISON_MODE else ""
        base_filename = f"osr_summary_{experiment_group_name}{fair_mode_suffix}_{timestamp}"

        csv_path = os.path.join(self.config.OSR_RESULT_DIR, f"{base_filename}.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.8f')
        print(f"\nOverall OSR results saved to CSV: {csv_path}")

        # Pivot tables 생성
        self._save_pivot_tables(results_df, self.config.OSR_RESULT_DIR, base_filename)

        osr_config_subset = { k: getattr(self.config, k) for k in dir(self.config) if k.startswith('OSR_') or k in ['RANDOM_STATE', 'EXPERIMENT_MODE', 'CURRENT_NLP_DATASET']}
        with open(os.path.join(self.config.OSR_RESULT_DIR, f"{base_filename}_config.json"), 'w') as f:
            json.dump({'osr_config': osr_config_subset, 'results_dataframe': results_df.to_dict(orient='records')}, f, indent=2, default=str)
        print(f"Overall OSR results and config saved to JSON.")

    def _save_pivot_tables(self, results_df: pd.DataFrame, save_dir: str, base_filename: str):
        """결과 DataFrame을 피벗 테이블로 변환하여 저장"""
        print("\n=== Creating Pivot Tables ===")
        
        # 피벗할 메트릭 목록
        metrics_to_pivot = ['AUROC_OOD', 'FPR@TPR95', 'OSCR', 'Closed_Set_Accuracy', 'F1_Macro']
        
        for metric in metrics_to_pivot:
            if metric not in results_df.columns:
                continue
                
            try:
                # 피벗 테이블 생성 (OOD 데이터셋을 행으로, OE 전략을 열로)
                pivot_df = results_df.pivot_table(
                    index='OOD_Dataset',
                    columns='OE_Source',
                    values=metric,
                    aggfunc='mean'  # 중복된 값이 있을 경우 평균
                )
                
                # 열 정렬 (Standard를 첫번째로, 나머지는 알파벳 순)
                if 'Standard' in pivot_df.columns:
                    other_cols = [col for col in pivot_df.columns if col != 'Standard']
                    ordered_cols = ['Standard'] + sorted(other_cols)
                    pivot_df = pivot_df[ordered_cols]
                
                # 행 정렬 (알파벳 순)
                pivot_df = pivot_df.sort_index()
                
                # 각 행과 열에 대한 평균 추가
                pivot_df['Mean'] = pivot_df.mean(axis=1)
                pivot_df.loc['Mean'] = pivot_df.mean(axis=0)
                
                # 최대/최소값 하이라이트를 위한 정보 추가 (선택적)
                if metric in ['AUROC_OOD', 'Closed_Set_Accuracy', 'F1_Macro', 'OSCR']:
                    # 높을수록 좋은 메트릭
                    best_per_ood = pivot_df.drop('Mean', axis=1).drop('Mean', axis=0).idxmax(axis=1)
                else:
                    # 낮을수록 좋은 메트릭 (FPR@TPR95)
                    best_per_ood = pivot_df.drop('Mean', axis=1).drop('Mean', axis=0).idxmin(axis=1)
                
                # 파일명 생성
                pivot_filename = f"{base_filename}_pivot_{metric}.csv"
                pivot_path = os.path.join(save_dir, pivot_filename)
                
                # CSV로 저장
                pivot_df.to_csv(pivot_path, float_format='%.4f')
                print(f"  - Saved pivot table for {metric}: {pivot_filename}")
                
                # 콘솔에도 출력 (선택적)
                print(f"\nPivot Table for {metric}:")
                print(pivot_df.round(4))
                print()
                
            except Exception as e:
                print(f"  - Error creating pivot table for {metric}: {e}")

    def run_full_pipeline(self):
        print(f"Starting Pipeline ({self.config.EXPERIMENT_MODE.upper()})...")
        start_time = datetime.now()

        # 종합 NLP 실험인 경우 특별 처리
        if self.config.EXPERIMENT_MODE == "nlp" and self.config.COMPREHENSIVE_NLP_EXPERIMENTS:
            print("🚀 Running COMPREHENSIVE NLP EXPERIMENTS...")
            self.run_stage5_osr_experiments()  # 종합 실험은 OSR 실험만 실행
        else:
            # 기존 파이프라인 실행
            self.run_stage1_model_training()

            self.model = None
            self.data_module = None
            clear_memory()

            df_with_attention = self.run_stage2_attention_extraction()

            self.model = None
            self.attention_analyzer = None
            clear_memory()

            df_with_metrics, features = self.run_stage3_oe_extraction(df_with_attention)

            self.model = None
            self.oe_extractor = None
            clear_memory()

            self.run_stage4_visualization(df_with_metrics, features)

            self.run_stage5_osr_experiments()

        end_time = datetime.now()
        self._print_final_summary(start_time, end_time)
        print(f"\nPipeline Complete! Total time: {end_time - start_time}")

    def _check_existing_model(self) -> bool:
        if not os.path.exists(self.config.MODEL_SAVE_DIR): return False
        return any(f.endswith('.ckpt') for f in os.listdir(self.config.MODEL_SAVE_DIR))

    def _load_existing_model(self, checkpoint_path: Optional[str] = None):
        if self.data_module is None: self.data_module = EnhancedDataModule(self.config); self.data_module.setup()
        if checkpoint_path is None:
            if os.path.exists(self.config.MODEL_SAVE_DIR):
                ckpts = [os.path.join(self.config.MODEL_SAVE_DIR, f) for f in os.listdir(self.config.MODEL_SAVE_DIR) if f.endswith('.ckpt')]
                if ckpts: checkpoint_path = max(ckpts, key=os.path.getmtime)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading base model from: {checkpoint_path}")
            try:
                self.model = EnhancedModel.load_from_checkpoint(
                    checkpoint_path, config=self.config, num_labels=self.data_module.num_labels,
                    label2id=self.data_module.label2id, id2label=self.data_module.id2label,
                    class_weights=self.data_module.class_weights
                )
                print("Base model loaded successfully!")
            except Exception as e: print(f"Error loading model: {e}"); self.model = None
        else: print("Warning: No model checkpoint found."); self.model = None

    def _load_best_model(self, checkpoint_callback: ModelCheckpoint):
        if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
            self._load_existing_model(checkpoint_path=checkpoint_callback.best_model_path)
        else:
            print("Warning: Best model path not found. Using current or latest model."); self._load_existing_model()

    def _load_attention_results(self) -> Optional[pd.DataFrame]:
        path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_attention_{self.config.EXPERIMENT_MODE}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'top_attention_words' in df.columns: df['top_attention_words'] = df['top_attention_words'].apply(safe_literal_eval)
            return df
        raise FileNotFoundError(f"Attention results not found: {path}")

    def _load_final_metrics_and_features(self) -> Tuple[Optional[pd.DataFrame], Optional[List[np.ndarray]]]:
        metrics_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"df_with_all_metrics_{self.config.EXPERIMENT_MODE}.csv")
        features_path = os.path.join(self.config.ATTENTION_DATA_DIR, f"extracted_features_{self.config.EXPERIMENT_MODE}.npy")
        df_metrics, features_arr_list = None, None
        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path)
            if 'top_attention_words' in df_metrics.columns: df_metrics['top_attention_words'] = df_metrics['top_attention_words'].apply(safe_literal_eval)
        if os.path.exists(features_path):
            loaded_features = np.load(features_path, allow_pickle=True)
            if loaded_features.ndim == 1 and isinstance(loaded_features[0], np.ndarray): features_arr_list = list(loaded_features)
            elif loaded_features.ndim == 2: features_arr_list = [row for row in loaded_features]
        return df_metrics, features_arr_list

    def _print_attention_samples(self, df: pd.DataFrame, num_samples: int = 3):
        if df is None or df.empty: return
        print(f"\n--- Attention Analysis Samples (Max {num_samples}) ---")
        sample_df = df.sample(min(num_samples, len(df)), random_state=self.config.RANDOM_STATE)
        for _, row in sample_df.iterrows():
            print("-" * 30)
            print(f"Original: {str(row.get(self.config.TEXT_COLUMN, 'N/A'))[:150]}...")
            print(f"Top Words: {row.get('top_attention_words', [])}")
            print(f"Masked:   {str(row.get(self.config.TEXT_COLUMN_IN_OE_FILES, 'N/A'))[:150]}...")

    def _print_final_summary(self, start_time: datetime, end_time: datetime):
        print(f"\n{'='*50}\nPIPELINE SUMMARY ({self.config.EXPERIMENT_MODE.upper()})\n{'='*50}")
        if self.config.EXPERIMENT_MODE == "nlp" and self.config.COMPREHENSIVE_NLP_EXPERIMENTS:
            print(f"🚀 COMPREHENSIVE NLP EXPERIMENTS completed!")
            print(f"Datasets tested: {self.config.NLP_DATASETS_FOR_COMPREHENSIVE}")
        print(f"Duration: {end_time - start_time}, Output Dir: {self.config.OUTPUT_DIR}")

# === Main Function ===
def main():
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Enhanced OE/OSR Pipeline with Comprehensive NLP Support")
    parser.add_argument('--mode', type=str, choices=['syslog', 'nlp'], default=Config.EXPERIMENT_MODE)
    parser.add_argument('--nlp_dataset', type=str, choices=list(Config.NLP_DATASETS.keys()), default=Config.CURRENT_NLP_DATASET)
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR)
    parser.add_argument('--skip_base_training', action='store_true')
    parser.add_argument('--skip_attention_extraction', action='store_true')
    parser.add_argument('--skip_oe_extraction', action='store_true')
    parser.add_argument('--skip_visualization', action='store_true')
    parser.add_argument('--skip_osr_experiments', action='store_true')
    parser.add_argument('--osr_eval_only', action='store_true', help="Skip OSR training and only run evaluation on pre-trained models.")
    parser.add_argument('--osr_force_retrain', action='store_true', help="Force retraining of OSR models even if they already exist.")
    parser.add_argument('--use_percentile', action='store_true', help="Use percentile method instead of the default elbow method for OE extraction.")
    parser.add_argument('--fair_comparison', action='store_true', help="Enable fair comparison mode by sampling all OE sets to the minimum TOKEN COUNT.")
    
    # 종합 NLP 실험 플래그
    parser.add_argument('--all_nlp_datasets', action='store_true', help="Run OSR experiments with ALL NLP datasets as ID datasets for comprehensive comparison.")
    parser.add_argument('--ood_sampling_strategy', 
                       choices=['original', 'min', 'fixed', 'proportional'], 
                       default='original',
                       help="Strategy for balancing OOD dataset sizes")
    parser.add_argument('--ood_sample_size', type=int, default=500,
                       help="Fixed sample size for 'fixed' strategy")
    parser.add_argument('--ood_proportional_ratio', type=float, default=0.1,
                       help="Proportion of ID size for 'proportional' strategy")
    args = parser.parse_args()

    Config.EXPERIMENT_MODE = args.mode
    Config.OUTPUT_DIR = args.output_dir
    Config.BASE_OUTPUT_DIR = args.output_dir  # 추가: 기본 출력 디렉토리 저장
    
    if args.mode == 'nlp':
        Config.CURRENT_NLP_DATASET = args.nlp_dataset
        Config.COMPREHENSIVE_NLP_EXPERIMENTS = args.all_nlp_datasets

    Config.STAGE_MODEL_TRAINING = not args.skip_base_training
    Config.STAGE_ATTENTION_EXTRACTION = not args.skip_attention_extraction
    Config.STAGE_OE_EXTRACTION = not args.skip_oe_extraction
    Config.STAGE_VISUALIZATION = not args.skip_visualization
    Config.STAGE_OSR_EXPERIMENTS = not args.skip_osr_experiments
    Config.OSR_EVAL_ONLY = args.osr_eval_only
    Config.OSR_FORCE_RETRAIN = args.osr_force_retrain
    Config.USE_ELBOW_METHOD = not args.use_percentile
    Config.OSR_FAIR_COMPARISON_MODE = args.fair_comparison

    # Re-initialize paths that depend on OUTPUT_DIR
    Config.MODEL_SAVE_DIR = os.path.join(Config.OUTPUT_DIR, "base_classifier_model")
    Config.LOG_DIR = os.path.join(Config.OUTPUT_DIR, "lightning_logs")
    Config.CONFUSION_MATRIX_DIR = os.path.join(Config.LOG_DIR, "confusion_matrices")
    Config.VIS_DIR = os.path.join(Config.OUTPUT_DIR, "oe_extraction_visualizations")
    Config.OE_DATA_DIR = os.path.join(Config.OUTPUT_DIR, "extracted_oe_datasets")
    Config.ATTENTION_DATA_DIR = os.path.join(Config.OUTPUT_DIR, "attention_analysis")
    Config.OSR_EXPERIMENT_DIR = os.path.join(Config.OUTPUT_DIR, "osr_experiments")
    Config.OSR_MODEL_DIR = os.path.join(Config.OSR_EXPERIMENT_DIR, "models")
    Config.OSR_RESULT_DIR = os.path.join(Config.OSR_EXPERIMENT_DIR, "results")

    Config.OOD_SAMPLING_STRATEGY = args.ood_sampling_strategy
    Config.OOD_SAMPLE_SIZE = args.ood_sample_size
    Config.OOD_PROPORTIONAL_RATIO = args.ood_proportional_ratio
    
    print(f"--- Pipeline Config ---")
    print(f"Mode: {Config.EXPERIMENT_MODE}")
    if Config.EXPERIMENT_MODE == 'nlp':
        if Config.COMPREHENSIVE_NLP_EXPERIMENTS:
            print(f"🚀 COMPREHENSIVE NLP EXPERIMENTS: Testing {Config.NLP_DATASETS_FOR_COMPREHENSIVE}")
        else:
            print(f"ID Dataset: {Config.CURRENT_NLP_DATASET}")
    else:
        print(f"ID Dataset: Syslog")
    print(f"Output: {Config.OUTPUT_DIR}")
    print(f"--- OE Extraction Method: {'Elbow Method' if Config.USE_ELBOW_METHOD else 'Percentile Method'} ---")
    if Config.OSR_FAIR_COMPARISON_MODE:
        print("--- OSR Experiment Mode: FAIR COMPARISON (by Token Budget) ---")
    else:
        print("--- OSR Experiment Mode: STANDARD (using original OE set sizes) ---")

    pipeline = EnhancedOEPipeline(Config)
    pipeline.run_full_pipeline()

if __name__ == '__main__':
    main()
