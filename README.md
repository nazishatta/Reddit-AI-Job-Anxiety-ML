#  Reddit AI Job Anxiety — ML Pipeline
### Predicting AI-Driven Job Anxiety from 446 GB of Reddit Discussions at Scale

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.x-orange?logo=apachespark)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-yellow?logo=amazonaws)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

##  Project Summary

This project builds an end-to-end machine learning pipeline to **detect and quantify AI-driven job anxiety** in Reddit discussions between **June 2023 and July 2024**.

Using **446 GB of Reddit comment data** processed on an **AWS EC2 Spark cluster**, the pipeline identifies comments where users discuss both AI/automation topics and job concerns, classifies whether anxiety is present, and predicts its intensity level.

> **This repository contains my individual ML contribution** to a larger group research project on AI's emotional and behavioral impact on online communities.

---

##  System Architecture

```
446 GB Reddit Parquet Data (Amazon S3)
         │
         ▼
┌─────────────────────────────┐
│   Stage 1: Apache Spark     │  ← Running on AWS EC2 cluster
│  - Read Phase-1 checkpoint  │
│  - Filter AI + Job keywords │
│  - Auto-label via weak      │
│    supervision              │
│  - Export labeled CSV → S3  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Stage 2: scikit-learn     │  ← sklearn ML pipeline
│  - TF-IDF feature pipeline  │
│  - Logistic Regression      │
│  - Linear SVM (calibrated)  │
│  - GridSearchCV tuning      │
│  - Threshold optimization   │
│  - Random + Temporal splits │
│  - Metrics → S3             │
└─────────────────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2b: Intensity Model  │
│  - 4-level anxiety scoring  │
│    (none / low / medium /   │
│     high)                   │
│  - Weighted severity tiers  │
└─────────────────────────────┘
```

---

##  What This Pipeline Does

### 1. Large-Scale Data Filtering (Spark)
- Reads **446 GB** of Reddit Parquet data from **Amazon S3**
- Filters 9 subreddits across 3 categories:
  - **Career:** `jobs`, `careerguidance`, `unemployment`
  - **AI-focused:** `MachineLearning`, `openai`, `datascience`
  - **Mental health:** `mentalhealth`, `Anxiety`, `depression`
- Applies **dual-keyword filtering** — comments must mention both AI terms AND job terms
- Uses **regex-based weak supervision** to auto-label anxiety presence

### 2. Smart Text Preprocessing
- URL removal and lowercasing
- **Negation handling** — phrases like *"not worried"* and *"don't fear"* are merged into single tokens before keyword matching, preventing false positives
- Custom `AnxietyKeywordMasker` — removes anxiety keywords from TF-IDF input to **prevent label leakage**

### 3. Feature Engineering Pipeline
Three parallel feature streams combined via `ColumnTransformer`:

| Stream | Features | Method |
|--------|----------|--------|
| Text | Up to 30,000 TF-IDF n-gram features (1–2 grams) | `TfidfVectorizer` |
| Metadata | Comment length | `MinMaxScaler` |
| Subreddit | Community context | `OneHotEncoder` |

### 4. Model Training & Evaluation
Two classifiers trained with full `GridSearchCV` hyperparameter tuning:

| Model | Tuned Parameters |
|-------|-----------------|
| Logistic Regression | `C` ∈ {0.1, 1.0, 10.0}, `class_weight` |
| Linear SVM (Calibrated) | `C` ∈ {0.1, 1.0, 5.0}, `class_weight` |

Two evaluation strategies:
- **Split A:** Random stratified 80/20 split
- **Split B:** Chronological 70/30 temporal split (train on older data → test on newer) to simulate real-world deployment

### 5. Threshold Optimization
- A validation slice is held out from training data
- The **optimal decision threshold** is selected by maximising F1 on the validation set, subject to a minimum recall floor
- This prevents the model from collapsing to always predict the majority class

### 6. Anxiety Intensity Scoring (Multi-class)
Beyond binary classification, the pipeline predicts **anxiety intensity** on 4 levels:

| Level | Score | Trigger |
|-------|-------|---------|
| None | 0 | No anxiety keywords |
| Low | 1 | 1 moderate keyword |
| Medium | 2–3 | Multiple moderate keywords |
| High | 4+ | Severe keywords (panic, hopeless, terrified, overwhelmed) |

---

##  Tech Stack

| Category | Tools |
|----------|-------|
| Big Data Processing | Apache Spark (PySpark), AWS EC2 cluster |
| Cloud Storage | Amazon S3 (boto3) |
| ML Framework | scikit-learn |
| Text Features | TF-IDF, custom transformers |
| Models | Logistic Regression, Linear SVM (CalibratedClassifierCV) |
| Tuning | GridSearchCV, StratifiedKFold |
| Evaluation | ROC-AUC, F1, Precision-Recall curves, confusion matrices |
| Visualization | Matplotlib |
| Language | Python 3.10 |

---

##  Repository Structure

```
reddit-ai-job-anxiety-ml/
├── src/
│   └── ml/
│       ├── ml_pipeline.py     ← Main end-to-end pipeline (Spark + sklearn)
│       └── config.py          ← All keywords, S3 paths, model parameters
├── outputs/
│   ├── plots/                 ← ROC curves, PR curves, confusion matrices
│   └── tables/                ← Metrics CSV, classification reports
└── README.md
```

---

##  How to Run

### Prerequisites
- Python 3.10+
- Apache Spark 3.x with S3A connector
- AWS credentials configured
- Access to S3 bucket with Phase-1 checkpoint

### Install dependencies
```bash
pip install pyspark scikit-learn pandas numpy matplotlib boto3
```

### Run the full pipeline
```bash
spark-submit \
  --master spark://172.31.83.150:7077 \
  --deploy-mode client \
  --driver-memory 3g \
  --executor-memory 2g \
  --executor-cores 1 \
  --num-executors 3 \
  src/ml/ml_pipeline.py
```

>  Stage 1 uses checkpointing — if the labeled CSV already exists on S3, it is downloaded automatically and Spark processing is skipped.

---

##  Key Engineering Decisions

**Why negation handling?**
Simple keyword matching would flag *"I'm not worried about AI"* as anxious. The pipeline merges negation phrases into single tokens (`not_worried`, `dont_worry`) before any downstream processing.

**Why mask anxiety keywords from TF-IDF?**
Labels are derived from anxiety keyword presence. Including those same keywords as TF-IDF features would create a trivial shortcut — the model would learn to detect the labeling rule rather than underlying linguistic patterns. The custom `AnxietyKeywordMasker` transformer removes them before vectorization.

**Why a temporal split in addition to random split?**
Random splits assume the future looks like the past. The temporal 70/30 split tests whether the model generalizes to genuinely unseen future data — a much harder and more realistic evaluation for a time-series setting like Reddit discussions.

**Why calibrated SVM?**
`LinearSVC` does not natively produce probabilities. Wrapping it in `CalibratedClassifierCV` enables probability estimates, which are required for threshold optimization and ROC-AUC evaluation.

---

##  Outputs Generated

| Output | Description |
|--------|-------------|
| `tables/metrics_summary.csv` | Accuracy, Precision, Recall, F1, ROC-AUC for all models and splits |
| `tables/label_distribution.csv` | Class balance statistics |
| `tables/report_*_random.csv` | Full classification reports |
| `plots/confusion_*_random.png` | Confusion matrices |
| `plots/roc_*_random.png` | ROC curves |
| `plots/pr_*_random.png` | Precision-Recall curves |
| `tables/feature_importance_lr.csv` | Top TF-IDF features by LR weight |
| `results_table.md` | Markdown summary table of all results |

---

##  Limitations & Honest Notes

- Labels are derived from **weak supervision** (keyword matching), not manual annotation. Models may partially learn the labeling heuristic rather than true anxiety.
- Class imbalance is handled via downsampling (3:1 negative:positive ratio) and `class_weight='balanced'`, but minority class performance remains a challenge.
- The temporal split training set may be small if date coverage is uneven — results from this split should be treated as indicative rather than definitive.
- Dataset is limited to English-language Reddit communities.

---

##  Project Context

This is the **ML component** of a larger group research project (*DATS 6450*) studying AI's emotional and behavioral impact on Reddit. The full project also includes an EDA component and an NLP component built by other team members. This repository contains only my individual contribution.
