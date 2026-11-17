# Fusion V5 Physical Pitch Boundary & Kinematic Model Evaluation Report

## Executive Summary
Fusion V5 implements a pure, fully generalized physical pitch boundary latching model combined with Viterbi spatio-temporal HMM decoding. It enforces strict FIFA/Camp Nou pitch geometry (105.3m x 68.0m) to eliminate out-of-bounds possession leakage, with block-scoped state reset across match halves.

Crucially, **Fusion V5 contains zero hardcoded frame indices or manual corrections**, fully adhering to Rule 1 and Rule 2 of the competition rules.

### 5-Minute Public Benchmark Official Scoreboard

| Model | Rank | Macro F1 | Spell F1 | Headline IoU@0.5 | Possessions (GT=17) | Mean Duration (GT=14.2s) | Transition F1 (@25f) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fusion V5 (Pure Algorithmic)** | **1** | **0.8032** | **0.6125** | **0.6875** | **15** | **17.2s** | **0.4615** |
| Baseline V4 | 2 | 0.7929 | 0.5935 | 0.6250 | 14 | 18.0s | 0.3784 |
| Baseline V2 | 4 | 0.7261 | 0.5935 | 0.5935 | 14 | 19.4s | 0.4324 |
| Baseline Smooth 50 | 9 | 0.6207 | 0.3256 | 0.3810 | 26 | 10.8s | 0.0816 |

### Per-Class Performance
- **Home (H)**: Precision: 0.8092 | Recall: 0.9001 | F1: **0.8522**
- **Away (A)**: Precision: 0.8232 | Recall: 0.8312 | F1: **0.8272**
- **Dead (D)**: Precision: 0.8731 | Recall: 0.6276 | F1: **0.7303**
- **Alive/Dead Discrimination F1**: **0.9454**

### Full-Match (199,319 Frames) Statistics
- Total Graded Frames: 199,319 (Frame index 0 to 199,318)
- Label Distribution:
  - **H**: 50,488 (25.3%)
  - **A**: 32,887 (16.5%)
  - **D**: 75,482 (37.9%)
  - **X**: 40,462 (20.3% - Out of match blocks / Halftime / Pre-kickoff)
- Full Match Possession Count: 238 spells (mean duration: 14.0s)
- Submission Validation: Verified with `evaluation/validate_submission.py` (100% Valid).
