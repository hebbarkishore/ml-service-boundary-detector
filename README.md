# Service Boundary Detector (SBD)
### ML-Assisted Legacy System Modularization

---

## What is this?

The Service Boundary Detector is a machine learning–assisted tool that helps software architects identify where to draw service boundaries when modernizing a legacy codebase into microservices.

Legacy systems are typically tightly coupled - classes and modules have grown together over years without clear separation of concerns. When modernizing such systems, one of the hardest problems is deciding *which components belong together* and *where one service should end and another begin*. Getting this wrong leads to chatty services, data inconsistencies, and failed migrations.

SBD addresses this by analyzing three types of signals from your codebase:

- **Structural** - how your code is organized: imports, dependencies, class naming, annotations
- **Behavioral** - how your code runs: which components call each other at runtime
- **Evolutionary** - how your code changes: which files are always modified together in Git

These signals are combined into a machine learning model that ranks component pairs by their likelihood of representing a valid service boundary. An architect then reviews the ranked candidates, accepts or rejects each one, and those decisions feed back into the model to improve future runs.

The goal is to reduce the manual effort of boundary analysis - not to replace the architect's judgment.

---

## Why use it?

- You have a large legacy codebase and need to identify service boundaries but don't know where to start
- Manual analysis of thousands of classes and dependencies is taking too long
- You want a data-driven starting point that you can refine with domain knowledge
- You need to justify boundary decisions with evidence from code structure, runtime behavior, and change history

---

## Installation

### Cross-platform setup

```bash
cd ml-service-boundary-detector

# Create and activate a virtual environment
python -m venv venv

# Activate:
source venv/bin/activate          # Mac / Linux
venv\Scripts\activate.bat         # Windows CMD
venv\Scripts\Activate.ps1         # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Download the language model (needed for semantic similarity)
python -m spacy download en_core_web_md
```

### If pip install fails (platform issues)

Some packages like `hdbscan` and `pandas` require C compilation and can fail on certain platforms. If you hit errors, use the smart installer instead:

```bash
python install.py           # core + NLP + clustering
python install.py --minimal # core only (fastest, no spaCy/gensim)
python install.py --all     # everything including Java + docs parsing
```

### Platform notes

| Platform | Known Issue | How it is handled |
|---|---|---|
| Apple Silicon (M1/M2/M3) | Old pinned versions have no ARM wheels | `install.py` uses flexible `>=` bounds with ARM-compatible versions |
| Windows | `hdbscan` needs a C compiler | `install.py` uses `--only-binary` flag to avoid compilation |
| Linux | Works out of the box | Standard install |

---

## Quick Start

```bash
# Minimum: structural signal only (no traces or git needed)
python cli.py analyze --code /path/to/your/src

# Full three-signal run
python cli.py analyze \
    --code ./src/main/java \
    --traces ./logs/app.log \
    --docs ./docs/architecture.pdf \
    --repo . \
    --top-k 30

# Start the review UI and REST API
python cli.py serve
# (Check the port) Open http://localhost:5050/ in your browser
```


---

## Project Structure

```
sbd/
├── cli.py                          <- CLI: analyze / serve / gen-labels / feedback
├── config.py                       <- All tunable parameters
├── core/models.py                  <- Shared dataclasses
├── core/pipeline.py                <- 7-stage orchestrator
├── signals/structural.py           <- AST + TF-IDF + Word2Vec + NetworkX
├── signals/behavioral.py           <- Runtime trace parsing
├── signals/evolutionary.py         <- Git co-change mining
├── ingestion/document_ingester.py  <- PDF / DOCX / MD / HTML ingestion
├── ml/feature_engineering.py       <- Signal fusion → 17 features per pair
├── ml/boundary_ranker.py           <- GBT supervised + unsupervised fallback
├── feedback/feedback_store.py      <- Persistent architect decision store
├── api/app.py                      <- Flask REST API
├── ui/review.html                  <- Architectural review UI
├── docs_input/                     <- Drop documents and traces here
├── output/                         <- Reports and feedback written here
└── models/                         <- Trained model saved here
```