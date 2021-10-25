"""
config.py  –  Central configuration for the ML Service Boundary Detector (ML-SBD).
All tunable knobs live here so nothing is buried in business logic.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR  = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"

for _d in (OUTPUT_DIR, MODEL_DIR, DOCS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


CODE_EXTENSIONS = {
    ".py":   "python",
    ".java": "java",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".go":   "go",
    ".cs":   "csharp",
}

DOC_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".html"}

# ── Structural Signal ──────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES   = 3000
TFIDF_NGRAM_RANGE    = (1, 2)
TFIDF_MIN_DF         = 2
COUPLING_STRONG_THR  = 5        # edge weight threshold for "strong" coupling

# ── Behavioral Signal (runtime trace parsing) ─────────────────────────────────
TRACE_LOG_PATTERNS = [
    # common Java/Python log formats; extend as needed
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})[^\[]*\[([^\]]+)\].*?(\w+\.\w+)\(",
    r"CALL\s+([\w.]+)\s+->\s+([\w.]+)",
    r"(\w[\w.]+)\s+calls\s+(\w[\w.]+)",
]

# ── Evolutionary Signal (git) ─────────────────────────────────────────────────
GIT_MAX_COMMITS      = 1000     # cap for large repos
GIT_SINCE_DAYS       = 365      # look-back window (1 year)
CO_CHANGE_MIN_COUNT  = 3        # minimum co-changes to form an edge

# ── ML / Ranking Model ────────────────────────────────────────────────────────
# Feature weights used to build the composite score when no labelled data
STRUCTURAL_WEIGHT    = 0.40
BEHAVIORAL_WEIGHT    = 0.35
EVOLUTIONARY_WEIGHT  = 0.25

# Supervised model hyperparams (GradientBoostingClassifier)
GB_N_ESTIMATORS      = 200
GB_MAX_DEPTH         = 4
GB_LEARNING_RATE     = 0.05
GB_SUBSAMPLE         = 0.8
RANDOM_STATE         = 42

# HDBSCAN clustering (used for unsupervised baseline)
HDBSCAN_MIN_CLUSTER_SIZE = 4
HDBSCAN_MIN_SAMPLES      = 3
HDBSCAN_METRIC           = "cosine"

# ── Candidate Pair Sampling ───────────────────────────────────────────────────
# Controls how many component-pairs are generated for ranking
MAX_PAIRS_PER_COMPONENT  = 50   # avoid O(n²) explosion on huge codebases

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5050
DEBUG    = os.getenv("SBD_DEBUG", "false").lower() == "true"

# ── spaCy Model ───────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_md"  
