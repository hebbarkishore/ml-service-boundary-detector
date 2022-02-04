import os
from pathlib import Path


BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR  = BASE_DIR / "models"
DOCS_DIR   = BASE_DIR / "docs_input"      
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


TFIDF_MAX_FEATURES   = 5000
TFIDF_NGRAM_RANGE    = (1, 2)
TFIDF_MIN_DF         = 2
COUPLING_STRONG_THR  = 5        

TRACE_LOG_PATTERNS = [
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})[^\[]*\[([^\]]+)\].*?(\w+\.\w+)\(",
    r"CALL\s+([\w.]+)\s+->\s+([\w.]+)",
    r"(\w[\w.]+)\s+calls\s+(\w[\w.]+)",
]


GIT_MAX_COMMITS      = 1000
GIT_SINCE_DAYS       = 365
CO_CHANGE_MIN_COUNT  = 3
EXECUTION_ORDER_MIN_CALLS = 3


STRUCTURAL_WEIGHT    = 0.40
BEHAVIORAL_WEIGHT    = 0.35
EVOLUTIONARY_WEIGHT  = 0.25


GB_N_ESTIMATORS      = 200
GB_MAX_DEPTH         = 4
GB_LEARNING_RATE     = 0.05
GB_SUBSAMPLE         = 0.8
RANDOM_STATE         = 42

HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES      = 2
HDBSCAN_METRIC           = "cosine"

MAX_PAIRS_PER_COMPONENT  = 50  



API_HOST = "0.0.0.0"
API_PORT = 5050
DEBUG    = os.getenv("SBD_DEBUG", "false").lower() == "true"

SPACY_MODEL = "en_core_web_md"   
