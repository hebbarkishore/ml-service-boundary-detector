# Service Boundary Detector (SBD)
### ML-Assisted Legacy System Modularization · Python · 

> Implements the three-signal ML framework described in:
> *ML-assisted approach for detecting service boundaries in legacy system modularization

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Service Boundary Detector                  │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │  STRUCTURAL │  │  BEHAVIORAL  │  │   EVOLUTIONARY    │   │
│  │             │  │              │  │                   │   │
│  │ AST parsing │  │ Trace log    │  │ Git co-change     │   │
│  │ Python/Java │  │ parsing      │  │ Zimmermann coeff  │   │
│  │ TF-IDF sim  │  │ Call freq    │  │ Decay weighting   │   │
│  │ Word2Vec    │  │ Temporal     │  │ Hotspot detection │   │
│  │ PageRank    │  │ co-occur     │  │                   │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘   │
│         │                │                   │               │
│         └────────────────┴───────────────────┘               │
│                          │                                   │
│               ┌──────────▼──────────┐                        │
│               │  Feature Engineering│                        │
│               │  (17 features/pair) │                        │
│               └──────────┬──────────┘                        │
│                          │                                   │
│               ┌──────────▼──────────┐                        │
│               │   ML Ranking Model  │                        │
│               │                     │                        │
│               │ Supervised:         │                        │
│               │  GradientBoosting   │                        │
│               │  + SMOTE balancing  │                        │
│               │                     │                        │
│               │ Unsupervised:       │                        │
│               │  Composite scoring  │                        │
│               │  + HDBSCAN clusters │                        │
│               └──────────┬──────────┘                        │
│                          │                                   │
│               ┌──────────▼──────────┐                        │
│               │  Ranked Boundary    │                        │
│               │  Candidates + Report│                        │
│               └─────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```
