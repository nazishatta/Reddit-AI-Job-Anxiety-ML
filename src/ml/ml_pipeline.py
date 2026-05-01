#!/usr/bin/env python3
"""
ML Pipeline — Reddit AI Job Anxiety Analysis
=============================================
Two-stage pipeline:
  Stage 1 (Spark)    : Read Phase-1 checkpoint → filter AI+job keywords
                       → auto-label → save labeled CSV to S3
  Stage 2 (sklearn)  : Train LR + LinearSVC on labeled CSV
                       → random split + temporal split
                       → save metrics, plots, confusion matrices to S3

Run (after Phase-1 checkpoint exists on S3):
  spark-submit --master spark://172.31.83.150:7077 \
    --deploy-mode client --driver-memory 3g \
    --executor-memory 2g --executor-cores 1 --num-executors 3 \
    src/ml/ml_pipeline.py
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType
    SPARK_AVAILABLE = True
except Exception:
    SPARK_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BUCKET           = "g21930448-nini-2026"
PHASE1_CHECKPOINT = f"s3a://{BUCKET}/outputs/ml/checkpoints/phase1_filtered/"
OUTPUT_LOCAL     = "outputs/ml/"
OUTPUT_S3_PREFIX = "outputs/ml/"
SAMPLE_SIZE      = 5000
RANDOM_STATE     = 42

NEGATIVE_RATIO   = 3     # negatives : positives in downsampled training set
VALIDATION_SIZE  = 0.20  # fraction of training data held out for threshold tuning
MIN_RECALL_FLOOR = 0.10  # ignore thresholds that push recall below this

# Informational only — Phase-1 checkpoint was already filtered by subreddit upstream.
# Not used for Spark filtering in Stage 1.
TARGET_SUBREDDITS = [
    "jobs", "careerguidance", "unemployment",
    "MachineLearning", "openai", "datascience",
    "mentalhealth", "Anxiety", "depression",
]

CAREER_SUBREDDITS = {"jobs", "careerguidance", "unemployment"}
AI_SUBREDDITS     = {"MachineLearning", "openai", "datascience"}
MENTAL_SUBREDDITS = {"mentalhealth", "Anxiety", "depression"}

AI_KEYWORDS = [
    r"\bai\b", r"\bartificial intelligence\b", r"\bchatgpt\b",
    r"\bgpt\b", r"\bllm\b", r"\bllms\b",
    r"\blarge language model\b", r"\blarge language models\b",
    r"\bgenerative ai\b", r"\bautomation\b", r"\bautomate\b",
    r"\bmachine learning\b",
]

JOB_KEYWORDS = [
    r"\bjob\b", r"\bjobs\b", r"\bcareer\b", r"\bcareers\b",
    r"\bwork\b", r"\bworker\b", r"\bworkers\b",
    r"\bemployment\b", r"\bunemployment\b", r"\bunemployed\b",
    r"\blayoff\b", r"\blayoffs\b", r"\blaid off\b",
    r"\breplace\b", r"\breplaced\b", r"\breplacing\b",
    r"\bhiring\b", r"\bprofession\b", r"\bprofessional\b",
    r"\breskill\b", r"\bupskill\b", r"\bfuture of work\b",
]

ANXIETY_KEYWORDS = [
    r"\banxiety\b", r"\banxious\b", r"\bworried\b", r"\bworry\b",
    r"\bscared\b", r"\bfear\b", r"\bfearful\b",
    r"\bstress\b", r"\bstressed\b", r"\bpanic\b",
    r"\boverwhelmed\b", r"\buncertain\b", r"\buncertainty\b",
    r"\bterrified\b", r"\bafraid\b", r"\bhopeless\b",
]

# Severity tiers for intensity scoring (subsets of ANXIETY_KEYWORDS)
SEVERE_ANXIETY_PATTERNS = [
    r"\bpanic\b", r"\boverwhelmed\b", r"\bterrified\b", r"\bhopeless\b",
]
MODERATE_ANXIETY_PATTERNS = [
    r"\banxiety\b", r"\banxious\b", r"\bworried\b", r"\bworry\b",
    r"\bscared\b", r"\bfear\b", r"\bfearful\b",
    r"\bstress\b", r"\bstressed\b",
    r"\buncertain\b", r"\buncertainty\b", r"\bafraid\b",
]
INTENSITY_LABELS = ["none", "low", "medium", "high"]

# Negated anxiety phrases — replaced with a merged token before keyword matching
# so "not worried" doesn't trigger the "worried" anxiety signal.
NEGATION_PATTERNS = [
    (r"\bnot\s+worried\b",   "not_worried"),
    (r"\bnot\s+scared\b",    "not_scared"),
    (r"\bnot\s+afraid\b",    "not_afraid"),
    (r"\bdon'?t\s+worry\b",  "dont_worry"),
    (r"\bno\s+fear\b",       "no_fear"),
    (r"\bnot\s+stressed\b",  "not_stressed"),
    (r"\bnot\s+anxious\b",   "not_anxious"),
    (r"\bnot\s+concerned\b", "not_concerned"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


_CAREER_LOWER  = {x.lower() for x in CAREER_SUBREDDITS}
_AI_LOWER      = {x.lower() for x in AI_SUBREDDITS}
_MENTAL_LOWER  = {x.lower() for x in MENTAL_SUBREDDITS}

def subreddit_type(s: str) -> str:
    s = s.lower() if isinstance(s, str) else ""
    if s in _CAREER_LOWER:  return "career"
    if s in _AI_LOWER:      return "ai"
    if s in _MENTAL_LOWER:  return "mental_health"
    return "other"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    for pattern, replacement in NEGATION_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_count(text: str, patterns: List[str]) -> int:
    if not isinstance(text, str):
        return 0
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def anxiety_severity_score(text: str) -> int:
    """Weighted severity: severe keyword = 2 pts, moderate keyword = 1 pt."""
    severe   = keyword_count(text, SEVERE_ANXIETY_PATTERNS)
    moderate = keyword_count(text, MODERATE_ANXIETY_PATTERNS)
    return severe * 2 + moderate


def intensity_bucket(score: int) -> int:
    """Map raw severity score → 0=none, 1=low, 2=medium, 3=high."""
    if score == 0:    return 0
    elif score == 1:  return 1
    elif score <= 3:  return 2
    else:             return 3


def upload_to_s3(local_path: str, s3_key: str) -> None:
    try:
        boto3.client("s3", region_name="us-east-1").upload_file(local_path, BUCKET, s3_key)
        log.info("Uploaded → s3://%s/%s", BUCKET, s3_key)
    except Exception as e:
        log.warning("S3 upload failed for %s: %s", local_path, e)


def save_csv(df: pd.DataFrame, filename: str) -> str:
    local = os.path.join(OUTPUT_LOCAL, filename)
    ensure_dir(Path(local).parent)
    df.to_csv(local, index=False)
    upload_to_s3(local, OUTPUT_S3_PREFIX + filename)
    return local


def save_fig(filename: str) -> None:
    ensure_dir(OUTPUT_LOCAL)
    local = os.path.join(OUTPUT_LOCAL, filename)
    plt.savefig(local, dpi=150, bbox_inches="tight")
    plt.close()
    upload_to_s3(local, OUTPUT_S3_PREFIX + filename)


def safe_roc_auc(y_true, scores) -> Optional[float]:
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def downsample_training_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Randomly drop negatives so the ratio is NEGATIVE_RATIO : 1."""
    pos_idx = y[y == 1].index.to_numpy()
    neg_idx = y[y == 0].index.to_numpy()
    n_neg_keep = min(len(neg_idx), len(pos_idx) * NEGATIVE_RATIO)
    rng = np.random.RandomState(RANDOM_STATE)
    neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    keep = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(keep)
    return X.loc[keep], y.loc[keep]


def choose_best_threshold(pipe: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Return the threshold that maximises F1 on X_val, subject to MIN_RECALL_FLOOR."""
    scores = get_scores(pipe, X_val)
    if scores is None:
        return 0.5
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_val, scores)
    best_f1, best_thresh = -1.0, 0.5
    for p, r, t in zip(prec_arr[:-1], rec_arr[:-1], thresholds):
        if r < MIN_RECALL_FLOOR:
            continue
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    log.info("    Threshold chosen: %.4f (val F1=%.4f)", best_thresh, best_f1)
    return best_thresh

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Spark: export + auto-label
# ─────────────────────────────────────────────────────────────────────────────

def create_spark() -> "SparkSession":
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark not available.")
    spark = (
        SparkSession.builder
        .appName("RedditML_AIJobAnxiety")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", "s3.us-east-1.amazonaws.com")
        .config("spark.hadoop.fs.s3a.endpoint.region", "us-east-1")
        .config("spark.hadoop.fs.s3a.bucket.probe", "0")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def stage1_export_and_label(spark: "SparkSession") -> str:
    """
    Read Phase-1 checkpoint, filter AI+job keywords, auto-label,
    sample SAMPLE_SIZE rows, save labeled CSV to outputs/ml/ and S3.
    Returns local path to labeled CSV.
    """
    labeled_csv = os.path.join(OUTPUT_LOCAL, "annotation_candidates_labeled.csv")

    # Skip if already done
    if os.path.exists(labeled_csv):
        log.info("Labeled CSV already exists — skipping Stage 1")
        return labeled_csv

    # Check S3
    try:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.head_object(Bucket=BUCKET, Key=OUTPUT_S3_PREFIX + "annotation_candidates_labeled.csv")
        log.info("Labeled CSV found on S3 — downloading")
        ensure_dir(OUTPUT_LOCAL)
        s3.download_file(BUCKET, OUTPUT_S3_PREFIX + "annotation_candidates_labeled.csv", labeled_csv)
        return labeled_csv
    except Exception:
        pass

    log.info("Stage 1: Reading Phase-1 checkpoint from %s", PHASE1_CHECKPOINT)
    df = spark.read.parquet(PHASE1_CHECKPOINT)

    # Cast partition columns if needed
    for col in ["yyyy", "mm"]:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(IntegerType()))

    # Clean text
    clean_col = F.trim(F.regexp_replace(
        F.regexp_replace(F.lower(F.col("body")), r"http\S+|www\S+", " "),
        r"\s+", " "
    ))
    df = df.withColumn("clean_body", clean_col)

    # Filter: must mention AI AND job keywords
    ai_pat  = "(" + "|".join(AI_KEYWORDS)  + ")"
    job_pat = "(" + "|".join(JOB_KEYWORDS) + ")"
    anx_pat = "(" + "|".join(ANXIETY_KEYWORDS) + ")"

    df = df.filter(F.col("clean_body").rlike(ai_pat))
    df = df.filter(F.col("clean_body").rlike(job_pat))

    # Auto-label: 1 = anxiety present, 0 = not
    df = df.withColumn(
        "label",
        F.when(F.col("clean_body").rlike(anx_pat), 1).otherwise(0)
    )

    total = df.count()
    log.info("Candidate pool: %s comments (AI + job related)", f"{total:,}")

    # Attempt balanced candidate sampling (up to SAMPLE_SIZE//2 per label class).
    # NOTE: labels are keyword-derived weak supervision, NOT hand-annotated ground truth.
    # If the positive pool is smaller than SAMPLE_SIZE//2, the final CSV will reflect
    # the natural class distribution rather than an artificially balanced one.
    half = SAMPLE_SIZE // 2
    pos = df.filter(F.col("label") == 1).orderBy(F.rand(RANDOM_STATE)).limit(half)
    neg = df.filter(F.col("label") == 0).orderBy(F.rand(RANDOM_STATE + 1)).limit(half)
    sample = pos.unionByName(neg).orderBy(F.rand(RANDOM_STATE + 2))

    pdf = sample.toPandas()
    pdf["subreddit_type"] = pdf["subreddit"].apply(subreddit_type)

    log.info("Label distribution:\n%s", pdf["label"].value_counts().to_string())
    log.info("Subreddit distribution:\n%s", pdf["subreddit"].value_counts().to_string())

    ensure_dir(OUTPUT_LOCAL)
    pdf.to_csv(labeled_csv, index=False)
    upload_to_s3(labeled_csv, OUTPUT_S3_PREFIX + "annotation_candidates_labeled.csv")
    log.info("Stage 1 complete — saved %s rows to %s", len(pdf), labeled_csv)

    return labeled_csv

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — sklearn: feature engineering + model training
# ─────────────────────────────────────────────────────────────────────────────

class AnxietyKeywordMasker(BaseEstimator, TransformerMixin):
    """Remove anxiety keywords from text before TF-IDF to prevent label leakage."""
    _pattern = re.compile(
        "|".join(ANXIETY_KEYWORDS), flags=re.IGNORECASE
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = pd.Series(X).fillna("").astype(str) if not isinstance(X, pd.Series) else X.fillna("").astype(str)
        return texts.apply(lambda t: self._pattern.sub(" ", t)).tolist()


class MetadataAdder(BaseEstimator, TransformerMixin):
    # Only text_len is kept — keyword counts are excluded to avoid shortcut signals
    # (all samples already passed the AI+job filter, making those counts near-redundant
    # and potentially correlated with the weak-supervision labeling signal).
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = pd.Series(X).fillna("").astype(str) if not isinstance(X, pd.Series) else X.fillna("").astype(str)
        return pd.DataFrame({"text_len": texts.str.len()})


def build_pipelines() -> Dict[str, Pipeline]:
    text_pipe = Pipeline([
        ("masker", AnxietyKeywordMasker()),
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            stop_words="english",
            min_df=2, max_df=0.95, ngram_range=(1, 2), max_features=30000,
        ))
    ])
    meta_pipe = Pipeline([
        ("meta",    MetadataAdder()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler",  MinMaxScaler()),
    ])
    sub_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("text",      text_pipe, "clean_body"),
        ("meta",      meta_pipe, "clean_body"),
        ("subreddit", sub_pipe,  ["subreddit"]),
    ], remainder="drop", sparse_threshold=0.3)

    return {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                C=1.0, solver="liblinear", random_state=RANDOM_STATE,
            )),
        ]),
        "linear_svm": Pipeline([
            ("preprocessor", preprocessor),
            ("model", CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
                cv=3,
            )),
        ]),
    }


PARAM_GRIDS = {
    "logistic_regression": {
        "model__C": [0.1, 1.0, 10.0],
        "model__class_weight": ["balanced", {0: 1, 1: 5}, {0: 1, 1: 10}],
    },
    # CalibratedClassifierCV wraps LinearSVC → params go through model__estimator__
    "linear_svm": {
        "model__estimator__C": [0.1, 1.0, 5.0],
        "model__estimator__class_weight": ["balanced", {0: 1, 1: 5}, {0: 1, 1: 10}],
    },
}


def tune_pipeline(pipe: Pipeline, param_grid: dict, X_train, y_train) -> Pipeline:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="roc_auc", n_jobs=-1, verbose=0, refit=True,
    )
    search.fit(X_train, y_train)
    log.info("    Best params: %s | CV AUC: %.4f", search.best_params_, search.best_score_)
    return search.best_estimator_


@dataclass
class EvalResult:
    model_name: str
    split_name: str
    accuracy:   float
    precision:  float
    recall:     float
    f1:         float
    roc_auc:    Optional[float]


def get_scores(pipe: Pipeline, X: pd.DataFrame) -> Optional[np.ndarray]:
    clf = pipe.named_steps["model"]
    try:
        if hasattr(clf, "predict_proba"):
            return pipe.predict_proba(X)[:, 1]
        if hasattr(clf, "decision_function"):
            transformed = pipe.named_steps["preprocessor"].transform(X)
            return clf.decision_function(transformed)
    except Exception as e:
        log.warning("get_scores failed for %s: %s", type(clf).__name__, e)
    return None


def evaluate(y_true, y_pred, scores, model_name, split_name) -> EvalResult:
    return EvalResult(
        model_name=model_name, split_name=split_name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=safe_roc_auc(y_true, scores) if scores is not None else None,
    )


def plot_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    plt.tight_layout()
    save_fig(f"plots/{filename}")


def plot_roc(y_true, scores, title, filename):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set(title=title, xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(f"plots/{filename}")


def plot_pr(y_true, scores, title, filename):
    prec, rec, _ = precision_recall_curve(y_true, scores)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2)
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1, label=f"Baseline ({baseline:.2f})")
    ax.set(title=title, xlabel="Recall", ylabel="Precision")
    ax.legend()
    plt.tight_layout()
    save_fig(f"plots/{filename}")



def save_feature_importance(pipe: Pipeline, filename: str, top_n: int = 25):
    clf = pipe.named_steps["model"]
    if not isinstance(clf, LogisticRegression):
        return
    try:
        preprocessor = pipe.named_steps["preprocessor"]
        # Access sub-components directly — avoids AnxietyKeywordMasker breaking the chain
        tfidf_names = preprocessor.named_transformers_["text"].named_steps["tfidf"].get_feature_names_out()
        sub_names   = preprocessor.named_transformers_["subreddit"].named_steps["onehot"].get_feature_names_out()
        feat_names  = np.concatenate([
            np.array([f"text__{n}"      for n in tfidf_names]),
            np.array(["meta__text_len"]),
            np.array([f"subreddit__{n}" for n in sub_names]),
        ])
        coefs = clf.coef_[0]
        rows = []
        for idx in np.argsort(coefs)[-top_n:][::-1]:
            rows.append({"feature": feat_names[idx], "coefficient": float(coefs[idx]), "direction": "anxiety"})
        for idx in np.argsort(coefs)[:top_n]:
            rows.append({"feature": feat_names[idx], "coefficient": float(coefs[idx]), "direction": "not_anxiety"})
        save_csv(pd.DataFrame(rows), f"tables/{filename}")
    except Exception as e:
        log.warning("Could not save feature importance: %s", e)


def build_intensity_pipelines() -> Dict[str, Pipeline]:
    """Multiclass intensity pipelines — anxiety keywords kept in TF-IDF (they are the signal)."""
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english",
            min_df=2, max_df=0.95, ngram_range=(1, 2), max_features=30000,
        ))
    ])
    meta_pipe = Pipeline([
        ("meta",    MetadataAdder()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler",  MinMaxScaler()),
    ])
    sub_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("text",      text_pipe, "clean_body"),
        ("meta",      meta_pipe, "clean_body"),
        ("subreddit", sub_pipe,  ["subreddit"]),
    ], remainder="drop", sparse_threshold=0.3)

    return {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                C=1.0, solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]),
        "linear_svm": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearSVC(
                C=1.0, class_weight="balanced",
                max_iter=2000, random_state=RANDOM_STATE,
            )),
        ]),
    }


def plot_intensity_confusion_matrix(cm, title, filename, class_names=None):
    labels = class_names or INTENSITY_LABELS
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, n + 2), max(4, n + 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Pred {l}" for l in labels], rotation=30, ha="right")
    ax.set_yticklabels([f"Actual {l}" for l in labels])
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_fig(f"plots/{filename}")


def stage2_intensity_train(labeled_csv: str) -> None:
    """
    Q2: Predict anxiety intensity (none/low/medium/high) using ordinal weak labels
    derived from anxiety keyword severity scoring (severe=2pts, moderate=1pt).
    Labels are weakly supervised approximations, not human-annotated ground truth.
    Reports macro F1, weighted F1, MAE (treating labels as ordinal), and confusion matrix.
    """
    log.info("Stage 2b: Intensity prediction (Q2) from %s", labeled_csv)

    df = pd.read_csv(labeled_csv)
    df["clean_body"] = df["body"].astype(str).map(clean_text)
    df = df[df["clean_body"].str.len() >= 10].copy()
    df["subreddit"] = df["subreddit"].str.lower().fillna("unknown")

    # Score on raw (lowercased) body — negation handling in clean_text strips
    # too many matches; raw scoring aligns with Stage-1 binary labeling logic.
    df["severity_score"]  = df["body"].astype(str).str.lower().apply(anxiety_severity_score)
    df["intensity_label"] = df["severity_score"].apply(intensity_bucket)

    # Collapse rare classes upward so every active class has ≥ 5 samples.
    # Merges high→medium, then medium→low if still too sparse.
    MIN_CLASS = 5
    for upper, lower in [(3, 2), (2, 1)]:
        if (df["intensity_label"] == upper).sum() < MIN_CLASS:
            df["intensity_label"] = df["intensity_label"].replace(upper, lower)
            log.info("  Merged intensity class %s→%s (too few samples)",
                     INTENSITY_LABELS[upper], INTENSITY_LABELS[lower])

    active_ids   = sorted(df["intensity_label"].unique())
    active_names = [INTENSITY_LABELS[i] for i in active_ids]

    dist = df["intensity_label"].value_counts().sort_index()
    log.info("Intensity label distribution (after merge): %s",
             {INTENSITY_LABELS[k]: int(v) for k, v in dist.items()})
    save_csv(
        pd.DataFrame({
            "intensity": [INTENSITY_LABELS[i] for i in dist.index],
            "label_id":  dist.index.tolist(),
            "count":     dist.values.tolist(),
        }),
        "tables/intensity_label_distribution.csv",
    )

    ensure_dir(os.path.join(OUTPUT_LOCAL, "plots"))
    ensure_dir(os.path.join(OUTPUT_LOCAL, "tables"))

    pipelines    = build_intensity_pipelines()
    all_results: List[dict] = []
    X = df[["clean_body", "subreddit"]]
    y = df["intensity_label"]

    # ── Split A: Random stratified ────────────────────────────────────────
    log.info("Intensity Split A: Random 80/20 stratified")
    min_class_n = int(y.value_counts().min())
    strat = y if min_class_n >= 2 else None
    if strat is None:
        log.warning("  Skipping stratify — a class has < 2 samples")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=strat, random_state=RANDOM_STATE
    )

    fitted_intensity: Dict[str, Pipeline] = {}
    for name, pipe in pipelines.items():
        log.info("  Training intensity %s (random)...", name)
        pipe.fit(X_tr, y_tr)
        fitted_intensity[name] = pipe
        y_pred = pipe.predict(X_te)

        mac_f1 = f1_score(y_te, y_pred, average="macro",    zero_division=0)
        wgt_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        mae    = float(np.mean(np.abs(y_te.values - y_pred.astype(int))))
        acc    = accuracy_score(y_te, y_pred)

        log.info("  %s random → Acc=%.4f MacroF1=%.4f WeightedF1=%.4f MAE=%.4f",
                 name, acc, mac_f1, wgt_f1, mae)
        all_results.append({
            "model_name": name, "split_name": "random_stratified",
            "accuracy": round(acc, 4), "macro_f1": round(mac_f1, 4),
            "weighted_f1": round(wgt_f1, 4), "mae": round(mae, 4),
        })

        cm = confusion_matrix(y_te, y_pred, labels=active_ids)
        plot_intensity_confusion_matrix(
            cm, f"Intensity: {name} — Random Split",
            f"intensity_confusion_{name}_random.png",
            active_names,
        )
        save_csv(
            pd.DataFrame(cm,
                index=[f"actual_{l}" for l in active_names],
                columns=[f"pred_{l}" for l in active_names]),
            f"tables/intensity_confusion_{name}_random.csv",
        )
        save_csv(
            pd.DataFrame(classification_report(
                y_te, y_pred, labels=active_ids,
                target_names=active_names, output_dict=True, zero_division=0,
            )).T,
            f"tables/intensity_report_{name}_random.csv",
        )

    # ── Split B: Temporal 70/30 ───────────────────────────────────────────
    if "yyyy" in df.columns and df["yyyy"].notna().any():
        temporal = df.dropna(subset=["yyyy"]).copy()
        temporal["yyyy"] = temporal["yyyy"].astype(int)
        temporal["mm"]   = pd.to_numeric(temporal.get("mm", 1), errors="coerce").fillna(1).astype(int)
        temporal = temporal.sort_values(["yyyy", "mm"]).reset_index(drop=True)
        split_idx = int(len(temporal) * 0.70)
        tr_t = temporal.iloc[:split_idx]
        te_t = temporal.iloc[split_idx:]
        log.info("Intensity Split B: Temporal 70/30 | train=%d test=%d", len(tr_t), len(te_t))

        for name, fitted_pipe in fitted_intensity.items():
            fresh = clone(fitted_pipe)
            fresh.fit(tr_t[["clean_body", "subreddit"]], tr_t["intensity_label"])
            y_pred = fresh.predict(te_t[["clean_body", "subreddit"]])
            y_true = te_t["intensity_label"].values

            mac_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
            wgt_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            mae    = float(np.mean(np.abs(y_true - y_pred.astype(int))))
            acc    = accuracy_score(y_true, y_pred)

            log.info("  %s temporal → Acc=%.4f MacroF1=%.4f WeightedF1=%.4f MAE=%.4f",
                     name, acc, mac_f1, wgt_f1, mae)
            all_results.append({
                "model_name": name, "split_name": "temporal_chronological_70_30",
                "accuracy": round(acc, 4), "macro_f1": round(mac_f1, 4),
                "weighted_f1": round(wgt_f1, 4), "mae": round(mae, 4),
            })

            cm = confusion_matrix(y_true, y_pred, labels=active_ids)
            plot_intensity_confusion_matrix(
                cm, f"Intensity: {name} — Temporal",
                f"intensity_confusion_{name}_temporal.png",
                active_names,
            )
            save_csv(
                pd.DataFrame(cm,
                    index=[f"actual_{l}" for l in active_names],
                    columns=[f"pred_{l}" for l in active_names]),
                f"tables/intensity_confusion_{name}_temporal.csv",
            )
            save_csv(
                pd.DataFrame(classification_report(
                    y_true, y_pred, labels=active_ids,
                    target_names=active_names, output_dict=True, zero_division=0,
                )).T,
                f"tables/intensity_report_{name}_temporal.csv",
            )

    # ── Summary ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results).round(4)
    save_csv(results_df, "tables/intensity_metrics_summary.csv")

    md = ["| Model | Split | Accuracy | Macro F1 | Weighted F1 | MAE |",
          "|---|---|---|---|---|---|"]
    for _, r in results_df.iterrows():
        md.append(
            f'| {r["model_name"]} | {r["split_name"]} | {r["accuracy"]:.4f} | '
            f'{r["macro_f1"]:.4f} | {r["weighted_f1"]:.4f} | {r["mae"]:.4f} |'
        )

    md_path = os.path.join(OUTPUT_LOCAL, "intensity_results_table.md")
    Path(md_path).write_text("\n".join(md))
    upload_to_s3(md_path, OUTPUT_S3_PREFIX + "intensity_results_table.md")

    log.info("\n=== INTENSITY RESULTS ===\n%s", results_df.to_string(index=False))
    log.info("Intensity outputs saved to %s and s3://%s/%s", OUTPUT_LOCAL, BUCKET, OUTPUT_S3_PREFIX)


def stage2_train(labeled_csv: str) -> None:
    # WEAK SUPERVISION NOTE: labels in this CSV are auto-generated via keyword rules,
    # not manually annotated. Models learn to detect contextual signals associated with
    # anxiety language, but results should be interpreted with that limitation in mind.
    # AUC is the primary metric; F1 is suppressed by class imbalance (~6% positive rate).
    log.info("Stage 2: Training models from %s", labeled_csv)

    df = pd.read_csv(labeled_csv)
    df["clean_body"] = df["body"].astype(str).map(clean_text)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df = df[df["clean_body"].str.len() >= 10].copy()
    df["subreddit"] = df["subreddit"].str.lower().fillna("unknown")
    label_counts = df["label"].value_counts().to_dict()
    log.info("Training on %s labeled rows | labels: %s", len(df), label_counts)
    pos = label_counts.get(1, 0)
    if pos < 200:
        log.warning(
            "LIMITATION: Only %d positive examples in training data. "
            "Class imbalance (~%.0f%% positive) will suppress F1/Recall. "
            "Use AUC as the primary evaluation metric. "
            "Report this as a limitation.", pos, 100 * pos / len(df)
        )
    # Compute single-keyword positive rate dynamically (not hardcoded).
    pos_kw_counts = df[df["label"] == 1]["body"].apply(
        lambda t: keyword_count(str(t), ANXIETY_KEYWORDS)
    )
    single_kw_pct = round(100 * (pos_kw_counts == 1).sum() / max(len(pos_kw_counts), 1))
    log.info(
        "Label quality: %d positives; %d%% have only 1 anxiety keyword (weak evidence). "
        "Class distribution approximates natural Reddit rate (Stage 1 could not oversample).",
        pos, single_kw_pct
    )

    ensure_dir(os.path.join(OUTPUT_LOCAL, "plots"))
    ensure_dir(os.path.join(OUTPUT_LOCAL, "tables"))

    # Save label distribution
    save_csv(df["label"].value_counts().reset_index().rename(columns={"index": "label", "label": "count"}),
             "tables/label_distribution.csv")
    save_csv(df.groupby(["subreddit", "label"]).size().reset_index(name="count"),
             "tables/label_distribution_by_subreddit.csv")

    pipelines  = build_pipelines()
    all_results: List[dict] = []

    X = df[["clean_body", "subreddit"]]
    y = df["label"]

    # ── Split A: Random stratified ────────────────────────────────────────
    log.info("Split A: Random 80/20 stratified")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Hold out a validation slice from training for threshold tuning (never touches test)
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=VALIDATION_SIZE, stratify=y_tr, random_state=RANDOM_STATE
    )
    # Downsample negatives in the training portion only
    X_tr2_ds, y_tr2_ds = downsample_training_data(X_tr2, y_tr2)
    log.info(
        "  Downsampled training: %d rows (%d pos / %d neg)",
        len(y_tr2_ds), int(y_tr2_ds.sum()), int((y_tr2_ds == 0).sum())
    )

    best_pipelines: Dict[str, Pipeline] = {}
    best_thresholds: Dict[str, float] = {}

    for name, pipe in pipelines.items():
        log.info("  Tuning %s (random split, downsampled)...", name)
        tuned = tune_pipeline(pipe, PARAM_GRIDS[name], X_tr2_ds, y_tr2_ds)
        best_pipelines[name] = tuned

        thresh = choose_best_threshold(tuned, X_val, y_val)
        best_thresholds[name] = thresh

        scores = get_scores(tuned, X_te)
        if scores is not None:
            y_pred = (scores >= thresh).astype(int)
        else:
            y_pred = tuned.predict(X_te)

        res = evaluate(y_te.values, y_pred, scores, name, "random_stratified")
        all_results.append(res.__dict__)
        log.info("  %s random → Acc=%.4f F1=%.4f AUC=%s (thresh=%.4f)",
                 name, res.accuracy, res.f1, f"{res.roc_auc:.4f}" if res.roc_auc else "N/A", thresh)

        cm = confusion_matrix(y_te, y_pred)
        plot_confusion_matrix(cm, f"{name} — Random Split", f"confusion_{name}_random.png")
        save_csv(pd.DataFrame(cm, index=["actual_0","actual_1"], columns=["pred_0","pred_1"]),
                 f"tables/confusion_{name}_random.csv")
        save_csv(pd.DataFrame(classification_report(y_te, y_pred, output_dict=True, zero_division=0)).T,
                 f"tables/report_{name}_random.csv")

        if scores is not None and res.roc_auc is not None:
            plot_roc(y_te.values, scores, f"{name} ROC (Random)", f"roc_{name}_random.png")
            plot_pr(y_te.values, scores, f"{name} PR Curve (Random)", f"pr_{name}_random.png")

        if name == "logistic_regression":
            save_feature_importance(tuned, "feature_importance_lr.csv")

    # ── Split B: Chronological 70/30 temporal split ──────────────────────
    # Uses yyyy+mm ordering — avoids the 329-row 2023-only training problem
    # where only 20 positives were available (too noisy for reliable estimates).
    if "yyyy" in df.columns and df["yyyy"].notna().any():
        temporal = df.dropna(subset=["yyyy"]).copy()
        temporal["yyyy"] = temporal["yyyy"].astype(int)
        temporal["mm"]   = pd.to_numeric(temporal.get("mm", 1), errors="coerce").fillna(1).astype(int)
        temporal = temporal.sort_values(["yyyy", "mm"]).reset_index(drop=True)
        split_idx = int(len(temporal) * 0.70)
        tr_t = temporal.iloc[:split_idx]
        te_t = temporal.iloc[split_idx:]
        log.info(
            "Split B: Chronological 70/30 | train=%d (%d pos) test=%d (%d pos)",
            len(tr_t), int(tr_t["label"].sum()), len(te_t), int(te_t["label"].sum())
        )
        if len(tr_t) < 500:
            log.warning(
                "LIMITATION: Temporal training set has only %d rows (%d positive). "
                "Temporal estimates will be noisy — treat as indicative, not definitive.",
                len(tr_t), int(tr_t["label"].sum())
            )

        for name, best_pipe in best_pipelines.items():
            log.info("  Training %s (temporal split, best params)...", name)
            fresh = clone(best_pipe)
            X_tr_t = tr_t[["clean_body", "subreddit"]]
            y_tr_t = tr_t["label"]

            # Validation slice from temporal training set for threshold tuning
            if len(y_tr_t) > 50 and y_tr_t.sum() >= 5:
                X_tr_t2, X_val_t, y_tr_t2, y_val_t = train_test_split(
                    X_tr_t, y_tr_t, test_size=VALIDATION_SIZE, stratify=y_tr_t,
                    random_state=RANDOM_STATE
                )
                X_tr_t2_ds, y_tr_t2_ds = downsample_training_data(X_tr_t2, y_tr_t2)
                fresh.fit(X_tr_t2_ds, y_tr_t2_ds)
                thresh_t = choose_best_threshold(fresh, X_val_t, y_val_t)
            else:
                fresh.fit(X_tr_t, y_tr_t)
                thresh_t = best_thresholds.get(name, 0.5)

            X_te_t = te_t[["clean_body", "subreddit"]]
            scores = get_scores(fresh, X_te_t)
            if scores is not None:
                y_pred = (scores >= thresh_t).astype(int)
            else:
                y_pred = fresh.predict(X_te_t)

            res = evaluate(te_t["label"].values, y_pred, scores, name, "temporal_chronological_70_30")
            all_results.append(res.__dict__)
            log.info("  %s temporal → Acc=%.4f F1=%.4f AUC=%s (thresh=%.4f)",
                     name, res.accuracy, res.f1, f"{res.roc_auc:.4f}" if res.roc_auc else "N/A", thresh_t)

            cm = confusion_matrix(te_t["label"], y_pred)
            plot_confusion_matrix(cm, f"{name} — Temporal (Chronological 70/30)", f"confusion_{name}_temporal.png")
            save_csv(pd.DataFrame(cm, index=["actual_0","actual_1"], columns=["pred_0","pred_1"]),
                     f"tables/confusion_{name}_temporal.csv")

            if scores is not None and res.roc_auc is not None:
                plot_roc(te_t["label"].values, scores, f"{name} ROC (Temporal)", f"roc_{name}_temporal.png")

    # ── Summary ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results).round(4)
    save_csv(results_df, "tables/metrics_summary.csv")

    md = ["| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |",
          "|---|---|---|---|---|---|---|"]
    for _, r in results_df.iterrows():
        roc = "" if pd.isna(r.get("roc_auc")) else f'{r["roc_auc"]:.4f}'
        md.append(f'| {r["model_name"]} | {r["split_name"]} | {r["accuracy"]:.4f} | '
                  f'{r["precision"]:.4f} | {r["recall"]:.4f} | {r["f1"]:.4f} | {roc} |')

    md_path = os.path.join(OUTPUT_LOCAL, "results_table.md")
    Path(md_path).write_text("\n".join(md))
    upload_to_s3(md_path, OUTPUT_S3_PREFIX + "results_table.md")

    log.info("\n=== FINAL RESULTS ===\n%s", results_df.to_string(index=False))
    log.info("All outputs saved to %s and s3://%s/%s", OUTPUT_LOCAL, BUCKET, OUTPUT_S3_PREFIX)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()
    log.info("=" * 60)
    log.info("  ML Pipeline — Reddit AI Job Anxiety Analysis")
    log.info("=" * 60)

    ensure_dir(OUTPUT_LOCAL)

    # Stage 1: Spark export + auto-label
    spark = create_spark()
    labeled_csv = stage1_export_and_label(spark)
    spark.stop()
    log.info("Spark stopped. Starting sklearn training...")

    # Stage 2: sklearn training
    stage2_train(labeled_csv)

    # Stage 2b: intensity prediction (Q2)
    stage2_intensity_train(labeled_csv)

    log.info("Pipeline complete in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
