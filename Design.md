# ML-Assisted Service Boundary Detection - System Design


## Overview

A machine learning–assisted framework that helps architects identify service boundaries in legacy systems. Rather than automating decomposition, it ranks boundary candidates by likelihood so architects can focus their review where it matters most.

---

## Architecture Flow

```
Legacy System Artifacts
         │
         ▼
┌─────────────────────────────────────┐
│           Data Collection           │
│                                     │
│  → Source Code Analysis             │
│  → Runtime Execution Data      ─────┼──► Feature Construction
│  → Version Control History          │
│                                     │
└─────────────────────────────────────┘
                  ▲
                  │ (Feedback for Refinement)
                  │
         ┌────────┴──────────────────────┐
         │      Accept / Reject Feedback │◄──── Architectural Review
         └────────┬──────────────────────┘
                  │ (Labeling for Retraining)
                  ▼
┌─────────────────────────────────────┐
│         Feature Engineering         │
│                                     │
│  · Structural Features              │
│  · Behavioral Features              │
│  · Evolutionary Features            │
└──────────────┬──────────────────────┘
               │
               ▼
   Supervised Learning Model (Tree-based)
               │
               ▼
       Confidence Scoring
               │
               ▼
   Ranked Boundary Candidates ──────► Architectural Review
```

---
