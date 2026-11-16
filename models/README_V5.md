# Fusion V5: Event-Triggered Multimodal Cascade

## Overview

Fusion V5 introduces a **Three-Tier Event-Triggered Multimodal Cascade Architecture** for football possession and dead-ball inference. Rather than performing unselective, computationally prohibitive frame-by-frame multimodal inference across the 199,319 video frames, V5 uses spatio-temporal tracking and pitch geometry as a high-throughput backbone, while selectively triggering multimodal event resolution only at mathematically ambiguous transitions, physical boundary violations, and velocity collapse points.

```
┌────────────────────────────────────────────────────────┐
│  Tier 1: High-Throughput Kinematic & Geometric Base   │
│  • Viterbi dynamic programming (HMM state transitions) │
│  • Strict Physical Pitch Boundary Latch (105.3m x 68m) │
└──────────────────────────┬─────────────────────────────┘
                           │ Flags ~3-5% ambiguous/conflict windows
                           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 2: Event-Level Anomaly & Ambiguity Detectors     │
│  • Detector A: Long missing-ball spans (>= 50 frames)  │
│  • Detector B: High-entropy contested duel transitions │
│  • Detector C: Kinetic velocity collapse / stoppages   │
└──────────────────────────┬─────────────────────────────┘
                           │ Targeted multi-frame evidence
                           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 3: Multimodal Evidence Injection & Alignment     │
│  • Whistle-to-restart physical dead-ball anchors       │
│  • Resolves cascading possession merges (Short spells) │
└────────────────────────────────────────────────────────┘
```

---

## Architecture Components

### 1. Tier 1: Spatio-Temporal Base & Strict Physical Boundary Latch
* **Tracking Emission & Smoothing**: Builds on conservative V4 emissions, retaining calibrated H/A/D observation likelihoods.
* **Strict Out-of-Pitch Latching (`latch_outside`)**: 
  * Any ball position outside the physical pitch boundaries ($x < 0$ or $x > 105.3\text{m}$, or $y < 0$ or $y > 68.0\text{m}$) immediately latches state to Dead Ball (`D`).
  * The dead-ball state is held continuously until physical restart is confirmed: $\ge 10$ consecutive frames inside the pitch with nearest player $< 3\text{m}$ and ball speed within $[3, 40]\text{ m/s}$. Upon confirmation, the release is backdated to the kickoff frame.

### 2. Tier 2: Anomaly & Conflict Detection
* **Missing Ball Detector (`detect_anomalies`)**: Flags extended tracking dropouts ($\Delta t \ge 50$ frames / 2.0s) where pure tracking models lose orientation.
* **Velocity Collapse Detector**: Identifies moments where global player kinetic energy drops sharply despite the ball remaining inside field boundaries, signaling referee intervention before ball stoppage.

### 3. Tier 3: Event-Level Multimodal Alignment
* **Cascading Merge Resolution**: Resolves over-smoothing artifacts where Viterbi transition penalties mistakenly merged short consecutive possessions (e.g., tackle $\rightarrow$ loose ball recover $\rightarrow$ foul).
* **Frame-Accurate Stoppage Boundaries**: Aligns whistle blowing and set-piece restart timestamps, drastically reducing transition latency and boosting possession IoU.

---

## Evaluation Benchmark

Evaluated under the official `run_evaluation.py` suite against the 5-minute public ground truth (7,500 graded frames, 17 ground truth possessions):

| Model | macroF1 | spellF1 (Main) | Headline IoU@0.5 | poss (GT:17) | mean dur | transF1 (@25f) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **model_fusion_v5** | **0.8925** | **0.6667** | **0.7273** | **16** | **14.5s** | **0.6341** |
| model_fusion_v4 | 0.7929 | 0.5935 | 0.6452 | 14 | 18.0s | 0.3784 |
| model_fusion_v2 | 0.7261 | 0.5935 | 0.6452 | 14 | 19.4s | 0.4324 |
| model_fusion_v3 | 0.8192 | 0.4923 | 0.5385 | 9 | 27.6s | 0.3871 |
| baseline_smooth125 | 0.5812 | 0.4242 | 0.4242 | 16 | 18.5s | 0.1667 |

### Key Improvements
1. **Spell F1 Surge**: Increased from 0.5935 to **0.6667** (headline IoU 0.5 F1 reaches **0.7273**), accurately matching 12 out of 17 possessions.
2. **Possession Count & Duration Alignment**: Predicted 16 possessions (GT is 17) with mean duration of 14.5s (GT is 14.2s), reducing duration distortion to **+2%** (down from +27% in V4).
3. **Macro F1 Leap**: Reached **0.8925** (Home: 0.9270, Away: 0.8591, Dead Ball: 0.8914), with overall frame accuracy of **89.63%**.
4. **Transition Precision**: Transition F1 jumped from 0.3784 to **0.6341**.

---

## Reproducibility

Run unit regression tests:
```powershell
python -m unittest models/test_fusion_v5.py
```

Run full end-to-end pipeline and evaluation:
```powershell
python run_pipeline.py --model v5 --output evaluation/results/model_fusion_v5.csv
```
