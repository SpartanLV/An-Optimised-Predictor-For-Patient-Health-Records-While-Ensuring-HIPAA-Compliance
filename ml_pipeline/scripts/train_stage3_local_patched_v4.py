#!/usr/bin/env python
"""
Stage 3 Training (SyntheaMass / SyntheticMass v2)

Multi-label disease prediction with:
  (1) CNN + BiLSTM deep model over tokenized patient timelines
  (2) Unsupervised Categorical HMM feature extractor over the same token ids
  (3) Trainable fusion MLP that combines deep probabilities + HMM features

This script is meant to be run offline to export a versioned "model bundle" for a Stage-3 inference microservice.

Typical usage (PowerShell):
  python train_stage3_local.py --data_dir "D:\SyntheaMass Data\v2\csv" --output_dir "output_stage3" --max_patients 50000

Notes about GPU on Windows:
  - TensorFlow >= 2.11 on *native Windows* is typically CPU-only.
  - For NVIDIA CUDA GPU training with TF 2.15+, use WSL2 + Linux TensorFlow (recommended),
    or switch frameworks (e.g., PyTorch), or use older/alternative backends (DirectML has limitations).

Author: generated from the user's notebook (CSE400 Final Draft V5) and patched for local execution.
"""

from __future__ import annotations

import os
import re
import json
import random
import warnings
import logging
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, Counter
from heapq import heappush, heappop
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import joblib

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

# HMM (hmmlearn)
try:
    from hmmlearn.hmm import CategoricalHMM  # hmmlearn>=0.3
except Exception as e:
    CategoricalHMM = None
    print("[WARN] Could not import CategoricalHMM. Install hmmlearn>=0.3. Error:", repr(e))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("stage3-train")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Config
# -----------------------------

@dataclass
class TableSpec:
    date_cols: list[str]      # Candidate date columns (case-insensitive). First match is used.
    token_cols: list[str]     # Columns used to build tokens (case-insensitive). VALUE is handled specially for observations.
    prefix: str               # Token prefix (e.g., 'med', 'obs', 'proc').

@dataclass
class CFG:
    # Data
    synthea_csv_dir: str
    output_dir: str = "output_stage3"
    chunksize: int = 200_000

    # Limit number of patients for local experiments (None = use all patients)
    max_patients: int | None = 50_000

    # Task definition: multi-label
    index_mode: str = "latest"      # "latest" or "first"
    target_top_k: int = 25
    min_label_count: int = 200
    use_condition_stop: bool = True
    include_index_day_events: bool = True
    lookback_days: int | None = 730  # 2 years

    # Token sequence construction
    max_events_per_patient: int = 250
    max_tokens_per_event: int = 10
    max_tokens_per_patient: int = 512
    event_sep_token: str = "evt_sep"

    # Feature tables (exclude 'conditions' to avoid leakage)
    feature_tables: dict[str, TableSpec] = field(default_factory=lambda: {
        "encounters": TableSpec(
            date_cols=["START", "DATE"],
            token_cols=["CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION", "TYPE"],
            prefix="enc",
        ),
        "observations": TableSpec(
            date_cols=["DATE"],
            token_cols=["CODE", "DESCRIPTION", "VALUE", "UNITS", "TYPE"],
            prefix="obs",
        ),
        "medications": TableSpec(
            date_cols=["START", "DATE"],
            token_cols=["CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"],
            prefix="med",
        ),
        "procedures": TableSpec(
            date_cols=["DATE", "START"],
            token_cols=["CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"],
            prefix="proc",
        ),
        "careplans": TableSpec(
            date_cols=["START", "DATE"],
            token_cols=["CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"],
            prefix="cp",
        ),
        "immunizations": TableSpec(
            date_cols=["DATE"],
            token_cols=["CODE", "DESCRIPTION"],
            prefix="imm",
        ),
        "allergies": TableSpec(
            date_cols=["START", "DATE"],
            token_cols=["CODE", "DESCRIPTION"],
            prefix="alg",
        ),
    })

    # Observation numeric binning (reduces vocab explosion from raw numeric values)
    obs_value_bins: list[float] = field(default_factory=lambda: [
        -np.inf, 0, 1, 5, 10, 20, 50, 100, 200, 500, np.inf
    ])

    # Deep model
    max_vocab: int = 60_000
    max_seq_len: int = 512
    embedding_dim: int = 192
    batch_size: int = 64
    epochs: int = 12
    lr: float = 3e-4
    label_smoothing: float = 0.01

    # HMM feature extractor
    hmm_states: int = 8
    hmm_n_iter: int = 40
    hmm_min_seq_len: int = 25
    hmm_train_max_sequences: int | None = 30_000
    hmm_train_max_len: int | None = 256
    hmm_feat_max_len: int | None = 512
    hmm_feat_use_length: bool = True

    # Fusion
    fusion_hidden: int = 128
    fusion_dropout: float = 0.25
    fusion_epochs: int = 10
    fusion_lr: float = 1e-3

    # Threshold tuning
    threshold_grid: list[float] = field(default_factory=lambda: [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)])

    # Recommendations
    rec_window_days: int = 30
    rec_top_n: int = 10

    # Splits
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42

    # Optional baseline
    run_tfidf_baseline: bool = False

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

# -----------------------------
# Utilities
# -----------------------------

def resolve_csv_path(csv_dir: str | Path, filename: str) -> Path:
    """Resolve a CSV filename inside csv_dir case-insensitively.

    Supports plain CSV as well as common compressed variants (.csv.gz, .csv.zip).
    """
    csv_dir = Path(csv_dir)
    direct = csv_dir / filename
    if direct.exists():
        return direct

    # Common compressed variants
    if filename.lower().endswith(".csv"):
        gz = csv_dir / (filename + ".gz")   # e.g., conditions.csv.gz
        zipf = csv_dir / (filename + ".zip") # e.g., conditions.csv.zip
        if gz.exists():
            return gz
        if zipf.exists():
            return zipf

    lower = filename.lower()
    for pattern in ("*.csv", "*.csv.gz", "*.csv.zip"):
        for p in csv_dir.glob(pattern):
            if p.name.lower() == lower:
                return p
    return direct  # may not exist

def iter_csv_chunks(path: Path, chunksize: int, dtype=str):
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        dtype=dtype,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        yield chunk

def normalize_pid(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()

def safe_to_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if hasattr(dt.dt, "tz_localize"):
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    return dt

def ci_find_col(columns: list[str], candidates: list[str]) -> str | None:
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None

_ALLOWED_CHARS = re.compile(r"[^a-z0-9:_\-\.\+]+")

def clean_token_value(x: str, max_len: int = 80) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = _ALLOWED_CHARS.sub("", s)
    s = s.strip("_")
    if not s:
        return ""
    return s[:max_len]

def norm_col_name(col: str) -> str:
    s = str(col).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def normalize_code(x: str, max_len: int = 40) -> str:
    return clean_token_value(x, max_len=max_len)

def make_token(prefix: str, col: str, raw_value: str) -> str:
    v = clean_token_value(raw_value)
    if not v:
        return ""
    return f"{prefix}_{norm_col_name(col)}:{v}"

def recency_token(event_date: pd.Timestamp, index_date: pd.Timestamp) -> str:
    try:
        d = int((index_date - event_date).days)
    except Exception:
        return ""
    if d < 0:
        return ""
    if d <= 7:
        return "time:0_7d"
    if d <= 30:
        return "time:8_30d"
    if d <= 90:
        return "time:31_90d"
    if d <= 365:
        return "time:91_365d"
    if d <= 730:
        return "time:1_2y"
    return "time:2y_plus"

def obs_value_bin_token(code_or_desc: str, val: str, bins: list[float]) -> str:
    x = pd.to_numeric(val, errors="coerce")
    if pd.isna(x):
        return ""
    b = None
    for i in range(len(bins) - 1):
        if bins[i] <= x < bins[i+1]:
            b = i
            break
    if b is None:
        return ""
    key = clean_token_value(code_or_desc, max_len=30) or "unknown"
    return f"obs_valbin:{key}_b{b}"

def binarize_probs(p: np.ndarray, thr: float) -> np.ndarray:
    return (p >= thr).astype(int)

def tune_threshold(y_true: np.ndarray, p: np.ndarray, grid: list[float], metric: str = "micro") -> dict:
    best = {"thr": None, "micro_f1": -1.0, "macro_f1": -1.0, "subset_acc": -1.0}
    for thr in grid:
        yhat = binarize_probs(p, thr)
        micro = f1_score(y_true, yhat, average="micro", zero_division=0)
        macro = f1_score(y_true, yhat, average="macro", zero_division=0)
        acc = accuracy_score(y_true, yhat)
        score = micro if metric == "micro" else macro
        if score > (best["micro_f1"] if metric == "micro" else best["macro_f1"]):
            best = {"thr": float(thr), "micro_f1": float(micro), "macro_f1": float(macro), "subset_acc": float(acc)}
    return best

# -----------------------------
# Step 1: Index date + multi-label targets from conditions.csv
# -----------------------------

def build_patient_index_and_multilabel_from_conditions(cfg: CFG):
    cond_path = resolve_csv_path(cfg.synthea_csv_dir, "conditions.csv")
    if not cond_path.exists():
        raise FileNotFoundError(f"conditions.csv not found in {cfg.synthea_csv_dir}")

    logger.info("Pass 1/2: scanning conditions.csv for index dates + label frequencies...")
    best_date: dict[str, pd.Timestamp] = {}
    code_counts: Counter = Counter()
    code_to_desc: dict[str, str] = {}

    for chunk in iter_csv_chunks(cond_path, cfg.chunksize):
        if chunk is None or chunk.empty:
            continue

        pid_col   = ci_find_col(chunk.columns.tolist(), ["PATIENT"])
        start_col = ci_find_col(chunk.columns.tolist(), ["START", "DATE"])
        code_col  = ci_find_col(chunk.columns.tolist(), ["CODE"])
        desc_col  = ci_find_col(chunk.columns.tolist(), ["DESCRIPTION", "DESC"])

        if not pid_col or not start_col or not code_col:
            raise ValueError(f"conditions.csv missing required columns. Found pid={pid_col}, start={start_col}, code={code_col}")

        cols = [pid_col, start_col, code_col] + ([desc_col] if desc_col else [])
        sub = chunk[cols].dropna(subset=[pid_col, start_col, code_col])
        if sub.empty:
            continue

        sub["pid"] = sub[pid_col].map(normalize_pid)
        sub = sub[sub["pid"].str.match(UUID_RE, na=False)]
        if sub.empty:
            continue

        sub["start_dt"] = safe_to_datetime(sub[start_col])
        sub = sub.dropna(subset=["start_dt"])
        if sub.empty:
            continue

        # index date per patient
        if cfg.index_mode == "first":
            per_dates = sub.groupby("pid")["start_dt"].min()
            for pid, dt in per_dates.items():
                prev = best_date.get(pid)
                if prev is None or dt < prev:
                    best_date[pid] = dt
        else:
            per_dates = sub.groupby("pid")["start_dt"].max()
            for pid, dt in per_dates.items():
                prev = best_date.get(pid)
                if prev is None or dt > prev:
                    best_date[pid] = dt

        # label frequencies by code
        codes = sub[code_col].map(normalize_code)
        codes = codes[codes.astype(bool)]
        if len(codes):
            code_counts.update(codes.value_counts().to_dict())

        # map code->description
        if desc_col:
            for c, d in zip(sub[code_col].values, sub[desc_col].values):
                cc = normalize_code(c)
                if not cc or cc in code_to_desc:
                    continue
                dd = str(d).strip()
                if dd:
                    code_to_desc[cc] = dd

    logger.info("Patients with >=1 condition: %d", len(best_date))
    logger.info("Unique condition codes seen: %d", len(code_counts))

    keep_codes = [c for c, _ in code_counts.most_common(cfg.target_top_k)]
    keep_codes_set = set(keep_codes)
    logger.info("Initial label space (top_k=%d): %d codes", cfg.target_top_k, len(keep_codes))

    patient_set = set(best_date.keys())
    logger.info("Pass 2/2: building per-patient ACTIVE label sets at index_date...")
    labels_by_pid: dict[str, set[str]] = defaultdict(set)

    for chunk in iter_csv_chunks(cond_path, cfg.chunksize):
        if chunk is None or chunk.empty:
            continue

        pid_col   = ci_find_col(chunk.columns.tolist(), ["PATIENT"])
        start_col = ci_find_col(chunk.columns.tolist(), ["START", "DATE"])
        stop_col  = ci_find_col(chunk.columns.tolist(), ["STOP", "END"])
        code_col  = ci_find_col(chunk.columns.tolist(), ["CODE"])
        if not pid_col or not start_col or not code_col:
            continue

        cols = [pid_col, start_col, code_col] + ([stop_col] if stop_col else [])
        sub = chunk[cols].dropna(subset=[pid_col, start_col, code_col])
        if sub.empty:
            continue

        sub["pid"] = sub[pid_col].map(normalize_pid)

        mask_pat = sub["pid"].isin(patient_set)
        if not mask_pat.any():
            continue
        sub = sub.loc[mask_pat].copy()

        sub["code"] = sub[code_col].map(normalize_code)
        sub = sub[sub["code"].isin(keep_codes_set)]
        if sub.empty:
            continue

        sub["start_dt"] = safe_to_datetime(sub[start_col])
        sub = sub.dropna(subset=["start_dt"])
        if sub.empty:
            continue

        if cfg.use_condition_stop and stop_col:
            sub["stop_dt"] = safe_to_datetime(sub[stop_col])
        else:
            sub["stop_dt"] = pd.NaT

        sub["index_dt"] = sub["pid"].map(best_date)
        sub = sub.dropna(subset=["index_dt"])
        if sub.empty:
            continue

        active = (sub["start_dt"] <= sub["index_dt"])
        if cfg.use_condition_stop and stop_col:
            active = active & (sub["stop_dt"].isna() | (sub["stop_dt"] >= sub["index_dt"]))

        sub = sub.loc[active, ["pid", "code"]]
        if sub.empty:
            continue

        grp = sub.groupby("pid")["code"].agg(lambda s: set(s))
        for pid, codes in grp.items():
            labels_by_pid[pid].update(codes)

    rows = []
    for pid, idx_dt in best_date.items():
        labs = labels_by_pid.get(pid, set())
        if not labs:
            continue
        rows.append({"pid": pid, "index_date": idx_dt, "labels": sorted(labs)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No patients left after building label sets. Consider increasing target_top_k or lowering min_label_count.")

    patient_counts = Counter()
    for labs in df["labels"]:
        patient_counts.update(labs)

    keep2 = [c for c, cnt in patient_counts.items() if cnt >= cfg.min_label_count]
    keep2 = sorted(keep2, key=lambda c: patient_counts[c], reverse=True)
    keep2 = keep2[: min(len(keep2), cfg.target_top_k)]
    keep2_set = set(keep2)

    df["labels"] = df["labels"].map(lambda labs: [c for c in labs if c in keep2_set])
    df = df[df["labels"].map(len) > 0].copy()

    patient_counts2 = Counter()
    for labs in df["labels"]:
        patient_counts2.update(labs)

    logger.info("After filtering: %d patients | %d labels", len(df), len(keep2))
    logger.info("Top labels (patient counts):\n%s",
                pd.Series(patient_counts2).sort_values(ascending=False).head(15).to_string())

    code_to_desc = {c: code_to_desc.get(c, "") for c in keep2}
    return df.reset_index(drop=True), code_to_desc

# -----------------------------
# Step 2: Build per-patient event tokens
# -----------------------------

def build_patient_tokens(cfg: CFG, patient_index_df: pd.DataFrame) -> pd.DataFrame:
    patient_set = set(patient_index_df["pid"].unique())
    index_date_map = dict(zip(patient_index_df["pid"], patient_index_df["index_date"]))
    labels_map = dict(zip(patient_index_df["pid"], patient_index_df["labels"]))

    heaps: dict[str, list[tuple[int, int, list[str]]]] = defaultdict(list)
    tie = 0

    lookback = None
    if cfg.lookback_days is not None:
        lookback = pd.Timedelta(days=int(cfg.lookback_days))

    for table_name, spec in cfg.feature_tables.items():
        csv_path = resolve_csv_path(cfg.synthea_csv_dir, f"{table_name}.csv")
        if not csv_path.exists():
            logger.warning("Skipping missing table: %s", csv_path)
            continue

        logger.info("Processing table=%s | path=%s", table_name, csv_path.name)

        for chunk in iter_csv_chunks(csv_path, cfg.chunksize):
            if chunk is None or chunk.empty:
                continue

            pid_col = ci_find_col(chunk.columns.tolist(), ["PATIENT"])
            date_col = ci_find_col(chunk.columns.tolist(), spec.date_cols)
            if not pid_col or not date_col:
                continue

            chunk_pid = chunk[pid_col].map(normalize_pid)
            mask_pat = chunk_pid.isin(patient_set)
            if not mask_pat.any():
                continue

            sub = chunk.loc[mask_pat].copy()
            sub["pid"] = chunk_pid[mask_pat]

            sub["event_date"] = safe_to_datetime(sub[date_col])
            sub["index_date"] = sub["pid"].map(index_date_map)
            sub = sub.dropna(subset=["event_date", "index_date"])
            if sub.empty:
                continue

            if cfg.include_index_day_events:
                sub = sub[sub["event_date"] <= sub["index_date"]]
            else:
                sub = sub[sub["event_date"] < sub["index_date"]]
            if sub.empty:
                continue

            if lookback is not None:
                sub = sub[sub["event_date"] >= (sub["index_date"] - lookback)]
                if sub.empty:
                    continue

            sub = sub.sort_values("event_date")

            col_actual = {}
            for raw_col in spec.token_cols:
                use_col = ci_find_col(sub.columns.tolist(), [raw_col])
                if use_col:
                    col_actual[raw_col] = use_col

            obs_code_col = col_actual.get("CODE")
            obs_desc_col = col_actual.get("DESCRIPTION")

            for row in sub.itertuples(index=False):
                pid = getattr(row, "pid")
                event_date = getattr(row, "event_date")
                index_date = getattr(row, "index_date")

                toks: list[str] = []

                tbin = recency_token(event_date, index_date)
                if tbin:
                    toks.append(tbin)

                toks.append(f"src:{spec.prefix}")

                for raw_col, use_col in col_actual.items():
                    val = getattr(row, use_col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue

                    if spec.prefix == "obs" and raw_col.upper() == "VALUE":
                        key = ""
                        if obs_code_col:
                            key = getattr(row, obs_code_col)
                        if (not key) and obs_desc_col:
                            key = getattr(row, obs_desc_col)
                        tok = obs_value_bin_token(key, val, cfg.obs_value_bins)
                        if tok:
                            toks.append(tok)
                        continue

                    tok = make_token(spec.prefix, raw_col, val)
                    if tok:
                        toks.append(tok)

                    if len(toks) >= cfg.max_tokens_per_event:
                        break

                if not toks:
                    continue

                tie += 1
                dt_ns = int(pd.Timestamp(event_date).value)
                heap = heaps[pid]
                heappush(heap, (dt_ns, tie, toks))
                if len(heap) > cfg.max_events_per_patient:
                    heappop(heap)

    rows = []
    empty = 0
    for r in patient_index_df.itertuples(index=False):
        pid = r.pid
        heap = heaps.get(pid, [])
        if not heap:
            empty += 1
            continue

        events = sorted(heap, key=lambda x: (x[0], x[1]))
        toks: list[str] = []
        for _, _, ev_toks in events:
            toks.extend(ev_toks)
            toks.append(cfg.event_sep_token)

        if len(toks) > cfg.max_tokens_per_patient:
            toks = toks[-cfg.max_tokens_per_patient:]

        text = " ".join(toks).strip()
        if not text:
            empty += 1
            continue

        rows.append({
            "pid": pid,
            "index_date": r.index_date,
            "labels": labels_map.get(pid, []),
            "tokens": toks,
            "text": text
        })

    df = pd.DataFrame(rows)
    logger.info("Built df_model: %s | dropped empty=%d", df.shape, empty)
    if not df.empty:
        logger.info("Mean tokens/patient: %.1f | max: %d", df["tokens"].map(len).mean(), df["tokens"].map(len).max())
    return df

# -----------------------------
# Deep model
# -----------------------------

class PadEmbeddingZeroer(tf.keras.callbacks.Callback):
    def __init__(self, layer_name="embed"):
        super().__init__()
        self.layer_name = layer_name

    def on_train_begin(self, logs=None):
        self._zero_pad()

    def on_epoch_end(self, epoch, logs=None):
        self._zero_pad()

    def _zero_pad(self):
        emb = self.model.get_layer(self.layer_name)
        w = emb.get_weights()
        if not w:
            return
        mat = w[0]
        mat[0] = 0.0
        emb.set_weights([mat])

class ValF1Multilabel(tf.keras.callbacks.Callback):
    def __init__(self, X_val, Y_val, grid=None, batch_size=64):
        super().__init__()
        self.X_val = X_val
        self.Y_val = Y_val
        self.batch_size = batch_size
        self.grid = grid if grid is not None else [0.5]
        self.best = -1.0
        self.best_thr = 0.5
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        tuned = tune_threshold(self.Y_val, p, self.grid, metric="micro")
        micro = tuned["micro_f1"]
        macro = tuned["macro_f1"]
        thr = tuned["thr"]
        if logs is not None:
            logs["val_micro_f1"] = micro
            logs["val_macro_f1"] = macro
        logger.info("Epoch %d | val_micro_f1=%.4f | val_macro_f1=%.4f | thr=%.2f",
                    epoch + 1, micro, macro, thr)
        if micro > self.best:
            self.best = micro
            self.best_thr = thr
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            logger.info("Restored best weights by val_micro_f1=%.4f (thr=%.2f)", self.best, self.best_thr)

def build_cnn_bilstm_multilabel(vocab_size: int, num_labels: int, cfg: CFG) -> tf.keras.Model:
    inp = layers.Input(shape=(cfg.max_seq_len,), name="token_ids")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=cfg.embedding_dim,
        mask_zero=False,
        name="embed"
    )(inp)

    x = layers.SpatialDropout1D(0.15)(x)

    x = layers.Conv1D(192, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.Conv1D(192, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)

    mx = layers.GlobalMaxPooling1D()(x)
    av = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([mx, av])

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)

    out = layers.Dense(num_labels, activation="sigmoid", name="sigmoid")(x)
    return models.Model(inp, out, name="cnn_bilstm_multilabel")

# -----------------------------
# HMM feature extractor
# -----------------------------

def padded_to_seq_list(X_seq: np.ndarray, min_len: int = 1) -> list[np.ndarray]:
    seqs = []
    for row in X_seq:
        s = row[row > 0]
        if len(s) >= min_len:
            seqs.append(s.astype(int))
    return seqs

def fit_unsupervised_hmm(X_train_seq: np.ndarray, vocab_size: int, cfg: CFG) -> CategoricalHMM:
    if CategoricalHMM is None:
        raise ImportError("CategoricalHMM not available. Install hmmlearn>=0.3.")

    seqs = padded_to_seq_list(X_train_seq, min_len=cfg.hmm_min_seq_len)
    logger.info("HMM training sequences kept (len >= %d): %d / %d", cfg.hmm_min_seq_len, len(seqs), len(X_train_seq))
    if not seqs:
        raise RuntimeError("No sequences long enough for HMM. Lower hmm_min_seq_len or increase max_tokens_per_patient.")

    if cfg.hmm_train_max_sequences is not None and len(seqs) > cfg.hmm_train_max_sequences:
        rng = np.random.default_rng(cfg.random_seed)
        idx = rng.choice(len(seqs), size=cfg.hmm_train_max_sequences, replace=False)
        seqs = [seqs[i] for i in idx]
        logger.info("Subsampled HMM sequences to: %d", len(seqs))

    if cfg.hmm_train_max_len is not None:
        seqs = [s[-cfg.hmm_train_max_len:] for s in seqs]

    X_cat = np.concatenate([s.reshape(-1, 1) for s in seqs], axis=0)
    lengths = [len(s) for s in seqs]

    hmm = CategoricalHMM(
        n_components=cfg.hmm_states,
        n_iter=cfg.hmm_n_iter,
        random_state=cfg.random_seed,
        verbose=False,
        n_features=vocab_size,
    )
    hmm.fit(X_cat, lengths)
    return hmm

def hmm_extract_features(hmm: CategoricalHMM, X_seq: np.ndarray, cfg: CFG) -> np.ndarray:
    n_states = hmm.n_components
    extra = 1 + (1 if cfg.hmm_feat_use_length else 0)
    feat_dim = extra + n_states
    feats = np.zeros((len(X_seq), feat_dim), dtype=np.float32)

    for i, row in enumerate(X_seq):
        s = row[row > 0].astype(int)
        if len(s) == 0:
            feats[i, 0] = -50.0
            if cfg.hmm_feat_use_length:
                feats[i, 1] = 0.0
            continue

        if cfg.hmm_feat_max_len is not None:
            s = s[-cfg.hmm_feat_max_len:]

        X = s.reshape(-1, 1)
        L = max(len(s), 1)

        try:
            ll = float(hmm.score(X))
            avg_ll = ll / L
            if not np.isfinite(avg_ll):
                avg_ll = -50.0
        except Exception:
            avg_ll = -50.0

        feats[i, 0] = avg_ll
        offset = 1
        if cfg.hmm_feat_use_length:
            feats[i, 1] = float(L)
            offset = 2

        try:
            gamma = hmm.predict_proba(X)
            occ = gamma.mean(axis=0)
        except Exception:
            st = hmm.predict(X)
            occ = np.bincount(st, minlength=n_states) / max(len(st), 1)

        # safety: guard against any NaNs/Infs from numerical issues
        occ = np.nan_to_num(occ, nan=0.0, posinf=0.0, neginf=0.0)
        s_occ = float(occ.sum())
        if s_occ > 0:
            occ = occ / s_occ
        else:
            occ = np.ones(n_states, dtype=np.float32) / n_states

        feats[i, offset:offset + n_states] = occ.astype(np.float32)

    return feats

# -----------------------------
# Fusion
# -----------------------------

def build_fusion_mlp(num_labels: int, hmm_feat_dim: int, cfg: CFG) -> tf.keras.Model:
    inp_deep = layers.Input(shape=(num_labels,), name="deep_probs")
    inp_hmm  = layers.Input(shape=(hmm_feat_dim,), name="hmm_feats")
    x = layers.Concatenate()([inp_deep, inp_hmm])
    x = layers.Dense(cfg.fusion_hidden, activation="relu")(x)
    x = layers.Dropout(cfg.fusion_dropout)(x)
    out = layers.Dense(num_labels, activation="sigmoid", name="sigmoid")(x)
    return models.Model([inp_deep, inp_hmm], out, name="fusion_mlp")

# -----------------------------
# Recommendations
# -----------------------------

def build_recommendations(cfg: CFG, df_model: pd.DataFrame, code_to_desc: dict[str, str]) -> dict:
    pid_set = set(df_model["pid"].unique())
    labels_map = dict(zip(df_model["pid"], df_model["labels"]))
    index_map  = dict(zip(df_model["pid"], df_model["index_date"]))

    window = pd.Timedelta(days=int(cfg.rec_window_days))

    rec_tables = {
        "medications": {"date_cols": ["START", "DATE"]},
        "procedures":  {"date_cols": ["DATE", "START"]},
        "careplans":   {"date_cols": ["START", "DATE"]},
    }

    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for tbl, spec in rec_tables.items():
        p = resolve_csv_path(cfg.synthea_csv_dir, f"{tbl}.csv")
        if not p.exists():
            logger.warning("Recommendation table missing: %s", p)
            continue

        logger.info("Building recommendations from table=%s", tbl)

        for chunk in iter_csv_chunks(p, cfg.chunksize):
            if chunk is None or chunk.empty:
                continue

            pid_col = ci_find_col(chunk.columns.tolist(), ["PATIENT"])
            date_col = ci_find_col(chunk.columns.tolist(), spec["date_cols"])
            if not pid_col or not date_col:
                continue

            chunk_pid = chunk[pid_col].map(normalize_pid)
            mask = chunk_pid.isin(pid_set)
            if not mask.any():
                continue

            sub = chunk.loc[mask].copy()
            sub["pid"] = chunk_pid[mask]
            sub["date"] = safe_to_datetime(sub[date_col])
            sub["index"] = sub["pid"].map(index_map)
            sub = sub.dropna(subset=["date", "index"])
            if sub.empty:
                continue

            sub = sub[(sub["date"] > sub["index"]) & (sub["date"] <= (sub["index"] + window))]
            if sub.empty:
                continue

            col_code = ci_find_col(sub.columns.tolist(), ["CODE"])
            col_desc = ci_find_col(sub.columns.tolist(), ["DESCRIPTION"])

            for row in sub.itertuples(index=False):
                pid = getattr(row, "pid")
                labs = labels_map.get(pid, [])
                if not labs:
                    continue

                code = normalize_code(getattr(row, col_code)) if col_code else ""
                desc = str(getattr(row, col_desc)).strip() if col_desc else ""
                item = code if code else clean_token_value(desc, max_len=80)
                if code and desc:
                    item = f"{code}|{desc}"

                if not item:
                    continue

                for lab in labs:
                    counts[lab][tbl][item] += 1

    out = {}
    for lab, tbls in counts.items():
        out[lab] = {}
        for tbl, ctr in tbls.items():
            out[lab][tbl] = [{"item": k, "count": int(v)} for k, v in ctr.most_common(cfg.rec_top_n)]
    return out

# -----------------------------
# Environment / GPU setup
# -----------------------------

def configure_tf_runtime(enable_mixed_precision: bool = False) -> dict:
    info = {
        "python": str(os.sys.version),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
    logger.info("Runtime: %s", info)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Enabled TF GPU memory growth.")
        except Exception as e:
            logger.warning("Could not set memory growth: %s", repr(e))

        if enable_mixed_precision:
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                logger.info("Enabled mixed precision (mixed_float16).")
            except Exception as e:
                logger.warning("Could not enable mixed precision: %s", repr(e))
    else:
        logger.warning("TensorFlow does NOT see a GPU in this environment. Training will be CPU-only.")

    return info

# -----------------------------
# Main training pipeline
# -----------------------------

def run_training(cfg: CFG, enable_mixed_precision: bool = False) -> Path:
    env_info = configure_tf_runtime(enable_mixed_precision=enable_mixed_precision)

    # Step 1
    patient_index_df, code_to_desc = build_patient_index_and_multilabel_from_conditions(cfg)

    if cfg.max_patients is not None and len(patient_index_df) > cfg.max_patients:
        patient_index_df = patient_index_df.sample(n=cfg.max_patients, random_state=cfg.random_seed).reset_index(drop=True)
        logger.info("Downsampled patient_index_df to max_patients=%d", cfg.max_patients)

    # Step 2
    df_model = build_patient_tokens(cfg, patient_index_df)
    if df_model.empty:
        raise RuntimeError("No usable patients after tokenization. Check lookback window or feature table availability.")

    # Step 3: splits + binarization + tokenization
    X = df_model["text"].values
    y_labels = df_model["labels"].values

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_labels)
    label_names = mlb.classes_
    num_labels = len(label_names)

    logger.info("Num labels: %d", num_labels)
    logger.info("Example labels: %s", label_names[:10])

    def make_stratify_keys(Y_bin: np.ndarray) -> np.ndarray:
        freqs = Y_bin.sum(axis=0)
        freqs = np.where(freqs == 0, 1, freqs)
        keys = np.zeros((Y_bin.shape[0],), dtype=int)
        for i, row in enumerate(Y_bin):
            pos = np.where(row == 1)[0]
            if len(pos) == 0:
                keys[i] = -1
            else:
                keys[i] = int(pos[np.argmin(freqs[pos])])
        return keys

    keys_all = make_stratify_keys(Y)

    try:
        X_train, X_tmp, Y_train, Y_tmp, k_train, k_tmp = train_test_split(
            X, Y, keys_all,
            test_size=(cfg.test_size + cfg.val_size),
            random_state=cfg.random_seed,
            stratify=keys_all
        )
    except Exception as e:
        logger.warning("Stratified split failed (%s). Falling back to random split.", repr(e))
        X_train, X_tmp, Y_train, Y_tmp = train_test_split(
            X, Y,
            test_size=(cfg.test_size + cfg.val_size),
            random_state=cfg.random_seed,
            shuffle=True
        )

    val_frac_of_tmp = cfg.val_size / (cfg.test_size + cfg.val_size)
    keys_tmp = make_stratify_keys(Y_tmp)

    try:
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_tmp, Y_tmp,
            test_size=(1 - val_frac_of_tmp),
            random_state=cfg.random_seed,
            stratify=keys_tmp
        )
    except Exception as e:
        logger.warning("Stratified split (val/test) failed (%s). Falling back to random split.", repr(e))
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_tmp, Y_tmp,
            test_size=(1 - val_frac_of_tmp),
            random_state=cfg.random_seed,
            shuffle=True
        )

    logger.info("Splits: train=%d | val=%d | test=%d", len(X_train), len(X_val), len(X_test))

    tokenizer = Tokenizer(num_words=cfg.max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    def texts_to_padded(texts: np.ndarray) -> np.ndarray:
        seq = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=cfg.max_seq_len, padding="post", truncating="post")

    X_train_seq = texts_to_padded(X_train)
    X_val_seq   = texts_to_padded(X_val)
    X_test_seq  = texts_to_padded(X_test)

    logger.info("Sequence shapes: train=%s val=%s test=%s", X_train_seq.shape, X_val_seq.shape, X_test_seq.shape)

    # Optional baseline
    if cfg.run_tfidf_baseline:
        tfidf = TfidfVectorizer(
            analyzer="word",
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            min_df=2,
            max_features=80_000,
        )
        X_train_t = tfidf.fit_transform(X_train)
        X_val_t   = tfidf.transform(X_val)
        X_test_t  = tfidf.transform(X_test)

        svm = OneVsRestClassifier(LinearSVC())
        svm.fit(X_train_t, Y_train)

        val_pred = svm.predict(X_val_t)
        test_pred = svm.predict(X_test_t)

        logger.info("TFIDF+OvR LinearSVC val subset-acc: %.4f | micro-F1: %.4f | macro-F1: %.4f",
                    accuracy_score(Y_val, val_pred),
                    f1_score(Y_val, val_pred, average="micro", zero_division=0),
                    f1_score(Y_val, val_pred, average="macro", zero_division=0))

        logger.info("TFIDF+OvR LinearSVC test subset-acc: %.4f | micro-F1: %.4f | macro-F1: %.4f",
                    accuracy_score(Y_test, test_pred),
                    f1_score(Y_test, test_pred, average="micro", zero_division=0),
                    f1_score(Y_test, test_pred, average="macro", zero_division=0))

    # Step 4: deep model
    vocab_size = min(cfg.max_vocab, len(tokenizer.word_index) + 1)
    tf.keras.backend.clear_session()
    deep_model = build_cnn_bilstm_multilabel(vocab_size, num_labels, cfg)
    try:
        loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg.label_smoothing)
    except TypeError:
        logger.warning("BinaryCrossentropy(label_smoothing=...) not supported; using default BCE.")
        loss_fn = tf.keras.losses.BinaryCrossentropy()

    deep_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss=loss_fn,
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc")],
    )

    callbacks = [
        PadEmbeddingZeroer("embed"),
        ValF1Multilabel(X_val_seq, Y_val, grid=cfg.threshold_grid, batch_size=cfg.batch_size),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.5, verbose=1),
    ]

    deep_model.fit(
        X_train_seq, Y_train,
        validation_data=(X_val_seq, Y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    p_train_deep = deep_model.predict(X_train_seq, batch_size=cfg.batch_size, verbose=0)
    p_val_deep   = deep_model.predict(X_val_seq,   batch_size=cfg.batch_size, verbose=0)
    p_test_deep  = deep_model.predict(X_test_seq,  batch_size=cfg.batch_size, verbose=0)

    deep_best = tune_threshold(Y_val, p_val_deep, cfg.threshold_grid, metric="micro")
    logger.info("Deep (val) best thr=%.2f | micro-F1=%.4f | macro-F1=%.4f | subset-acc=%.4f",
                deep_best["thr"], deep_best["micro_f1"], deep_best["macro_f1"], deep_best["subset_acc"])

    # Step 5: HMM features
    hmm = fit_unsupervised_hmm(X_train_seq, vocab_size=vocab_size, cfg=cfg)

    hmm_train = hmm_extract_features(hmm, X_train_seq, cfg)
    hmm_val   = hmm_extract_features(hmm, X_val_seq,   cfg)
    hmm_test  = hmm_extract_features(hmm, X_test_seq,  cfg)

    # Safety: StandardScaler can't handle inf/nan (can happen if HMM scores -inf)
    # (e.g., if some tokens are unseen by the HMM training subset)
    hmm_train = np.nan_to_num(hmm_train, nan=0.0, posinf=0.0, neginf=-50.0).astype(np.float32)
    hmm_val   = np.nan_to_num(hmm_val,   nan=0.0, posinf=0.0, neginf=-50.0).astype(np.float32)
    hmm_test  = np.nan_to_num(hmm_test,  nan=0.0, posinf=0.0, neginf=-50.0).astype(np.float32)

    hmm_scaler = StandardScaler()
    hmm_train_s = hmm_scaler.fit_transform(hmm_train)
    hmm_val_s   = hmm_scaler.transform(hmm_val)
    hmm_test_s  = hmm_scaler.transform(hmm_test)

    # Step 6: Fusion model
    fusion_model = build_fusion_mlp(num_labels=num_labels, hmm_feat_dim=hmm_train_s.shape[1], cfg=cfg)
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.fusion_lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc")]
    )

    fusion_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.5, verbose=1),
    ]

    fusion_model.fit(
        [p_train_deep, hmm_train_s], Y_train,
        validation_data=([p_val_deep, hmm_val_s], Y_val),
        epochs=cfg.fusion_epochs,
        batch_size=cfg.batch_size,
        callbacks=fusion_callbacks,
        verbose=1,
    )

    p_val_fused  = fusion_model.predict([p_val_deep,  hmm_val_s],  batch_size=cfg.batch_size, verbose=0)
    p_test_fused = fusion_model.predict([p_test_deep, hmm_test_s], batch_size=cfg.batch_size, verbose=0)

    fused_best = tune_threshold(Y_val, p_val_fused, cfg.threshold_grid, metric="micro")
    logger.info("FUSED (val) best thr=%.2f | micro-F1=%.4f | macro-F1=%.4f | subset-acc=%.4f",
                fused_best["thr"], fused_best["micro_f1"], fused_best["macro_f1"], fused_best["subset_acc"])

    thr = fused_best["thr"]
    Y_test_hat = binarize_probs(p_test_fused, thr)

    test_subset_acc = accuracy_score(Y_test, Y_test_hat)
    test_micro_f1 = f1_score(Y_test, Y_test_hat, average="micro", zero_division=0)
    test_macro_f1 = f1_score(Y_test, Y_test_hat, average="macro", zero_division=0)

    logger.info("FUSED (test) subset-acc: %.4f | micro-F1: %.4f | macro-F1: %.4f", test_subset_acc, test_micro_f1, test_macro_f1)

    print("\nMulti-label classification report (test):")
    print(classification_report(Y_test, Y_test_hat, target_names=label_names, zero_division=0))

    # Step 7: recommendations
    recommendations = build_recommendations(cfg, df_model, code_to_desc)

    # Step 8: export model bundle
    bundle_id = datetime.utcnow().strftime("stage3_bundle_%Y%m%dT%H%M%SZ")
    bundle_dir = Path(cfg.output_dir) / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    deep_path = bundle_dir / "deep_cnn_bilstm_multilabel.keras"
    fusion_path = bundle_dir / "fusion_mlp.keras"
    deep_model.save(deep_path)
    fusion_model.save(fusion_path)

    joblib.dump(tokenizer, bundle_dir / "tokenizer.joblib")
    joblib.dump(mlb, bundle_dir / "multilabel_binarizer.joblib")
    joblib.dump(hmm, bundle_dir / "hmm_feature_extractor.joblib")
    joblib.dump(hmm_scaler, bundle_dir / "hmm_feature_scaler.joblib")

    (bundle_dir / "label_code_to_description.json").write_text(json.dumps(code_to_desc, indent=2))
    (bundle_dir / "recommendations.json").write_text(json.dumps(recommendations, indent=2))

    metadata = {
        "bundle_id": bundle_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "task": "multilabel_active_conditions_at_index_date",
        "labels": {
            "num_labels": int(num_labels),
            "label_names": list(label_names),
            "top_k": int(cfg.target_top_k),
            "min_label_count": int(cfg.min_label_count),
            "uses_condition_stop": bool(cfg.use_condition_stop),
            "index_mode": cfg.index_mode,
        },
        "tokenization": {
            "include_index_day_events": bool(cfg.include_index_day_events),
            "lookback_days": cfg.lookback_days,
            "max_events_per_patient": int(cfg.max_events_per_patient),
            "max_tokens_per_patient": int(cfg.max_tokens_per_patient),
            "max_tokens_per_event": int(cfg.max_tokens_per_event),
            "event_sep_token": cfg.event_sep_token,
            "obs_value_bins": cfg.obs_value_bins,
            "max_seq_len": int(cfg.max_seq_len),
            "max_vocab": int(cfg.max_vocab),
        },
        "models": {
            "deep_model_path": deep_path.name,
            "fusion_model_path": fusion_path.name,
            "hmm_states": int(cfg.hmm_states),
            "hmm_n_iter": int(cfg.hmm_n_iter),
            "fusion_hidden": int(cfg.fusion_hidden),
            "fusion_dropout": float(cfg.fusion_dropout),
            "fusion_lr": float(cfg.fusion_lr),
        },
        "thresholds": {
            "deep_best_val": deep_best,
            "fused_best_val": fused_best,
            "use_for_inference": {"thr": fused_best["thr"], "tuned_on": "val", "metric": "micro_f1"},
        },
        "metrics": {
            "fused_test": {
                "subset_acc": float(test_subset_acc),
                "micro_f1": float(test_micro_f1),
                "macro_f1": float(test_macro_f1),
            }
        },
    }

    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (bundle_dir / "cfg.json").write_text(json.dumps(asdict(cfg), indent=2))
    (bundle_dir / "env.json").write_text(json.dumps(env_info, indent=2))

    logger.info("Bundle exported to: %s", bundle_dir)
    return bundle_dir

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 training on SyntheaMass/SyntheticMass v2")
    p.add_argument("--data_dir", required=True, help="Folder containing Synthea CSVs (conditions.csv, encounters.csv, ...)")
    p.add_argument("--output_dir", default="output_stage3", help="Where to write the exported bundle")
    p.add_argument("--max_patients", type=int, default=50_000, help="Max patients for local training (0 or negative => all)")
    p.add_argument("--run_tfidf_baseline", action="store_true", help="Run TF-IDF + OvR LinearSVC baseline (can be slow/memory heavy)")
    p.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision if a GPU is visible to TF (WSL2/Linux)")
    # Quick knobs (optional)
    p.add_argument("--epochs", type=int, default=None, help="Override deep model epochs")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    p.add_argument("--target_top_k", type=int, default=None, help="Override target_top_k")
    p.add_argument("--min_label_count", type=int, default=None, help="Override min_label_count")
    return p.parse_args()

def main():
    args = parse_args()
    max_patients = args.max_patients
    if max_patients is not None and max_patients <= 0:
        max_patients = None

    cfg = CFG(
        synthea_csv_dir=args.data_dir,
        output_dir=args.output_dir,
        max_patients=max_patients,
        run_tfidf_baseline=args.run_tfidf_baseline,
    )

    # Auto-detect if the user passed a parent folder (e.g., contains a "csv" subfolder).
    def _auto_detect_synthea_csv_dir(p: str) -> str:
        base = Path(p)
        if base.exists() and base.is_file():
            base = base.parent

        def has_conditions_csv(d: Path) -> bool:
            return any((d / fn).exists() for fn in ("conditions.csv", "conditions.csv.gz", "conditions.csv.zip"))

        if has_conditions_csv(base):
            return str(base)

        candidates = [
            base / "csv",
            base / "output" / "csv",
            base / "synthea" / "output" / "csv",
            base / "synthea-master" / "output" / "csv",
        ]
        for c in candidates:
            if has_conditions_csv(c):
                return str(c)

        return str(base)

    cfg.synthea_csv_dir = _auto_detect_synthea_csv_dir(cfg.synthea_csv_dir)

    # Optional overrides
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.target_top_k is not None:
        cfg.target_top_k = int(args.target_top_k)
    if args.min_label_count is not None:
        cfg.min_label_count = int(args.min_label_count)

    logger.info("CFG: %s", cfg)

    out = run_training(cfg, enable_mixed_precision=bool(args.mixed_precision))
    print("DONE. Exported bundle:", out)

if __name__ == "__main__":
    main()
