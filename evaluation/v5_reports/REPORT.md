# V5 Multimodal Cascade Possession Model Evaluation Report

## Executive Summary
Fusion V5 implements an event-triggered multimodal cascade architecture that builds upon spatial tracking and strict pitch-geometry latching, combined with multimodal event refinement for high-entropy transitions.

### 5-Minute Public Benchmark Official Scoreboard

| Model | Rank | Macro F1 | Spell F1 | Headline IoU@0.5 | Possessions (GT=17) | Mean Duration (GT=14.2s) | Transition F1 (@25f) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fusion V5** | **1** | **0.8925** | **0.6667** | **0.7273** | **16** | **14.5s** | **0.6341** |
| Fusion V4 | 2 | 0.7929 | 0.5935 | 0.6250 | 14 | 18.0s | 0.3784 |
| Fusion V2 | 4 | 0.7261 | 0.5935 | 0.5935 | 14 | 19.4s | 0.4324 |
| Baseline Smooth 50 | 9 | 0.6207 | 0.3256 | 0.3810 | 26 | 10.8s | 0.0816 |

### Per-Class Performance
- **Home (H)**: Precision: 0.9554 | Recall: 0.9001 | F1: **0.9270**
- **Away (A)**: Precision: 0.8627 | Recall: 0.8556 | F1: **0.8591**
- **Dead (D)**: Precision: 0.8331 | Recall: 0.9585 | F1: **0.8914**
- **Alive/Dead Discrimination F1**: **0.9710**

### Full-Match (199,319 Frames) Statistics
- Total Graded Frames: 199,319 (Frame index 0 to 199,318)
- Label Distribution:
  - **H**: 49,900 (25.0%)
  - **A**: 32,841 (16.5%)
  - **D**: 76,116 (38.2%)
  - **X**: 40,462 (20.3% - Out of match blocks / Halftime / Pre-kickoff)
- Full Match Possession Count: 239 spells (mean duration: 13.8s)
- Submission Validation: Verified with `evaluation/validate_submission.py` (100% Valid).
