# Service Boundary Detector (SBD)

ML-assisted tool for identifying service boundaries in legacy codebases.

> This tool is the reference implementation for the research paper:
> **Machine Learning-Assisted Service Boundary Detection for Modularizing Legacy Systems**
> Kishore Subramanya Hebbar - IJAET Vol. 4 No. 2, 2022
> Full paper: https://romanpub.com/resources/ijaet-v4-2-2022-48.pdf
>
> Latest Code release: https://codeberg.org/kishorehebbar/ml-service-boundary-detector/releases

---

## What it does

When modernizing a large legacy system into microservices, the hardest question is: *where do the boundaries go?* Getting this wrong leads to over-chatty services, shared-database anti-patterns, and failed migrations.

SBD analyzes three complementary signals from your codebase and ranks component pairs by how likely they are to represent a valid service boundary:

- **Structural** - static dependencies, imports, class naming, annotations
- **Behavioral** - runtime call patterns from execution traces or application logs
- **Evolutionary** - co-change history from Git commits

It is a **decision-support tool**, not an automated decomposition engine. An architect reviews the ranked candidates, accepts or rejects each one, and those decisions improve future runs.

---

## Requirements

- Python 3.10, 3.11, or 3.12
- Git (for evolutionary signal extraction)
- Java source analysis requires `javalang` (installed automatically)

---

## Installation

```bash
git clone https://codeberg.org/kishorehebbar/ml-service-boundary-detector.git
cd ml-service-boundary-detector

python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate.bat       # Windows

pip install -r requirements.txt
python -m spacy download en_core_web_md
```

If `pip install` fails due to platform-specific compilation issues (common with `hdbscan` on Windows or older Macs):

```bash
python install.py           # recommended
python install.py --minimal # structural signals only, no spaCy / gensim
python install.py --all     # includes Java parsing and document ingestion
```

| Platform | Known issue | How handled |
|---|---|---|
| Apple Silicon | None | Standard install |
| Windows | `hdbscan` needs a C compiler | `install.py` uses `--only-binary` |
| Linux | None | Standard install |

---

## Quick Start

```bash
# Structural signals only — no traces or Git required
python cli.py analyze --code /path/to/src

# All three signals
python cli.py analyze \
    --code ./src/main/java \
    --traces ./logs/app.log \
    --repo . \
    --top-k 30

# Open the review UI
python cli.py serve
# Visit http://localhost:5050
```

---

## Workflow

1. **Analyze** — run `cli.py analyze` against your codebase
2. **Review** — open the web UI, work through the ranked boundary candidates
3. **Feedback** — accept or reject each pair; decisions are stored automatically
4. **Retrain** — on the next run, stored feedback is injected and the model improves

With fewer than 20 labelled pairs the ranker uses an unsupervised weighted composite score. Once enough feedback accumulates it switches to a supervised Gradient Boosting model trained on your decisions.

---

## Project Structure

```
ml-service-boundary-detector/
├── cli.py                          CLI: analyze / serve / gen-labels / feedback
├── config.py                       All tunable parameters
├── core/
│   ├── models.py                   Shared dataclasses 
│   └── pipeline.py                 
├── signals/
│   ├── structural.py               AST parsing, TF-IDF, Word2Vec, dependency graph
│   ├── behavioral.py               Trace parsing, execution order stability
│   └── evolutionary.py             Git co-change mining, sequence directionality
├── ingestion/
│   └── document_ingester.py        PDF / DOCX / Markdown / HTML ingestion
├── ml/
│   ├── feature_engineering.py      Signal fusion 
│   └── boundary_ranker.py          GBT supervised + adaptive unsupervised fallback
├── feedback/
│   └── feedback_store.py           Persists architect decisions across runs
├── api/
│   └── app.py                      Flask REST API
├── ui/
│   └── review.html                 Browser-based review interface
├── tests/                          Unit tests  behavioral, evolutionary, features
├── docs_input/                     Drop architecture docs and trace files here
├── output/                         Reports and feedback stored here
└── models/                         Trained model persisted here
```

---

## CLI Reference

```bash
python cli.py analyze --help
python cli.py serve --help
python cli.py feedback stats
python cli.py feedback list
python cli.py feedback import --file output/labels.json
python cli.py feedback clear
python cli.py gen-labels          
```

---

## Supported Languages

Python, Java, JavaScript, TypeScript, Go, C#

Trace log formats: log4j, Python logging, OpenTelemetry CSV, Spring HTTP
