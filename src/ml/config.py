"""
ML Pipeline Configuration
Project: Analyzing AI Impact on Job & Mental Health Discussions on Reddit
"""

# ─────────────────────────────────────────────
# S3 Paths
# ─────────────────────────────────────────────
BUCKET = "g21930448-nini-2026"
COMMENTS_PATH   = f"s3a://{BUCKET}/reddit-data/parquet/comments/"
OUTPUT_S3       = f"s3a://{BUCKET}/outputs/ml/"
OUTPUT_LOCAL    = "outputs/ml/"

# Checkpoint paths (skip phase if already done)
CHECKPOINT = {
    "filtered":  f"s3a://{BUCKET}/outputs/ml/checkpoints/phase1_filtered/",
    "labeled":   f"s3a://{BUCKET}/outputs/ml/checkpoints/phase2_labeled/",
    "features":  f"s3a://{BUCKET}/outputs/ml/checkpoints/phase3_features/",
}

# ─────────────────────────────────────────────
# Target Subreddits
# ─────────────────────────────────────────────
CAREER_SUBS     = ["jobs", "careerguidance", "unemployment"]
AI_SUBS         = ["MachineLearning", "openai", "datascience"]
MENTAL_SUBS     = ["mentalhealth", "Anxiety", "depression"]
ALL_SUBREDDITS  = CAREER_SUBS + AI_SUBS + MENTAL_SUBS

SUBREDDIT_TYPE  = (
    {s: "career"  for s in CAREER_SUBS} |
    {s: "ai"      for s in AI_SUBS}     |
    {s: "mental"  for s in MENTAL_SUBS}
)

# ─────────────────────────────────────────────
# Keyword Lists
# ─────────────────────────────────────────────
AI_KEYWORDS = [
    "chatgpt", "gpt-4", "gpt4", "gpt", "openai", "llm", "llms",
    "large language model", "artificial intelligence", " ai ", "a.i.",
    "machine learning", "automation", "automate", "automated",
    "generative ai", "bard", "gemini", "claude", "copilot",
    "ai tool", "ai replace", "ai job", "robot", "algorithm",
    "deep learning", "neural network", "midjourney", "stable diffusion",
    "ai model", "language model", "ai system",
]

ANXIETY_KEYWORDS = [
    "worried", "worry", "scared", "terrified", "anxious", "anxiety",
    "fear", "afraid", "panic", "stressed", "stress",
    "losing my job", "lose my job", "lost my job", "job loss",
    "replaced by ai", "replace workers", "replace humans",
    "laid off", "layoff", "lay off", "getting fired", "got fired",
    "unemployment", "unemployed", "no job", "can't find work",
    "can't find a job", "struggling to find", "hopeless", "desperate",
    "doom", "doomed", "obsolete", "useless", "worthless",
    "no future", "what's the point", "give up", "can't compete",
    "depressing", "terrifying", "nightmare", "will i lose",
    "is my job safe", "am i going to lose", "should i be worried",
    "took my job", "taking jobs", "take our jobs",
]

OPTIMISM_KEYWORDS = [
    "excited", "exciting", "opportunity", "opportunities",
    "reskill", "upskill", "adapt", "adapting",
    "growing", "growth", "promising", "optimistic", "hopeful",
    "future is bright", "new skills", "new opportunities",
    "can help", "will help", "productivity", "efficient",
    "augment", "collaborate", "partnership", "empower",
    "embrace", "looking forward", "positive", "benefit",
    "advantage", "create jobs", "new roles", "evolve",
]

# ─────────────────────────────────────────────
# Model Parameters
# ─────────────────────────────────────────────
TFIDF_VOCAB_SIZE    = 20000
TFIDF_MIN_DF        = 5
MAX_COMMENT_TOKENS  = 200

LR_MAX_ITER         = 100
LR_REG_PARAM        = 0.01

SVM_MAX_ITER        = 100
SVM_REG_PARAM       = 0.01

RANDOM_SEED         = 42
TRAIN_RATIO         = 0.8

# Temporal split: train on 2023, test on 2024
TRAIN_YEAR          = 2023
TEST_YEAR           = 2024
