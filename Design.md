# ML-Assisted Service Boundary Detection - System Design


## Overview

A machine learning-assisted framework that helps architects identify service boundaries in legacy systems. 

The system is designed as a **decision-support tool**, not an automation tool. Human judgment remains central to all final boundary decisions.

---

## Pipeline Stages

```
Legacy System Artifacts
        │
        ▼
┌───────────────────────────────────────────┐
│  Stage 1 · Data Collection                │
│                                           │
│  Source Code ──► Structural Signals       │  
│  Trace Logs  ──► Behavioral Signals       │
│  Git History ──► Evolutionary Signals     │
└───────────────────────┬───────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│  Stage 2 · Feature Construction           │  
│                                           │
│                                           │
│  Structural                               │
│    coupling weight, TF-IDF similarity,    │
│    semantic similarity, shared imports,   │
│    shared annotations, inheritance link   │
│                                           │
│  Behavioral                               │
│    call frequency, call depth,            │
│    temporal affinity,                     │
│    execution order stability              │
│                                           │
│  Evolutionary                             │
│    co-change frequency, recency,          │
│    logical coupling,                      │
│    change sequence directionality         │
│                                           │
│  Graph Centrality                         │
│                                           │
│                                           │
│  Cross-layer flag                         │
└───────────────────────┬───────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│  Stage 3 · Boundary Scoring               │  
│                                           │
│  ≥20 labelled pairs                       │
│    -> Supervised GBT (GradientBoosting    │
│      + SMOTE + StratifiedKFold CV)        │
│                                           │
│  < 20 labelled pairs                      │
│    -> Unsupervised weighted composite     │
│      with adaptive signal weights         │
│      (weights shift toward whichever      │
│       signal best separates accepted vs   │
│       rejected feedback decisions)        │
└───────────────────────┬───────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│  Stage 4 · Architectural Review           │  
│                                           │
│  Ranked candidates -> Web UI / JSON report│
│  Architect accepts or rejects each pair   │
│  Decisions persist in feedback store      │
│  Sufficient feedback -> model retraining  │
└───────────────────────────────────────────┘
```

---

## Signal Design

| Signal Group | Source | Key Insight (from paper) |
|---|---|---|
| Structural | AST + TF-IDF + Word2Vec | Compile-time coupling; insufficient alone — accidental dependencies mislead |
| Behavioral | Runtime traces / logs | Reveals actual execution paths; call frequency, **execution order stability** distinguish core from incidental dependencies |
| Evolutionary | Git commit history | Co-change patterns encode functional cohesion; **change sequence directionality** separates causal from coincidental edits |

---

## Adaptive Signal Weights

In unsupervised mode, the three channel weights (structural / behavioral / evolutionary) start at their configured base values. Once enough architect feedback accumulates, the weights shift toward whichever signal best discriminates accepted from rejected boundaries.



---

## Module Map

```
ml-service-boundary-detector/
│
├── core/
│   ├── models.py          PairFeatures, BoundaryCandidate, CodeUnit
│   └── pipeline.py        orchestrator
│
├── signals/
│   ├── structural.py      AST parsing, TF-IDF, Word2Vec, dependency graph
│   ├── behavioral.py      Trace parsing, execution order stability
│   └── evolutionary.py    Git mining, co-change decay, sequence directionality
│
├── ml/
│   ├── feature_engineering.py   Builds feature vectors per pair
│   └── boundary_ranker.py       GBT (supervised) + adaptive composite (unsupervised)
│
├── feedback/
│   └── feedback_store.py  Persists architect decisions; feeds retraining
│
├── api/
│   └── app.py             Flask REST API
│
├── ui/
│   └── review.html        Architect review interface
│
├── tests/
│   ├── test_behavioral.py          
│   ├── test_evolutionary.py        
│   └── test_feature_engineering.py 
│
├── cli.py                 CLI entry point (analyze / serve / feedback)
└── config.py              All tunable parameters
```
