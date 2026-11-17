# Team MSN: Event-Triggered Multimodal Cascade Possession Inference

**Challenge**: More Than A Hack 2027 (Barça Innovation Hub / Metrica Sports)  
**Deliverable**: Technical Document  
**Submission File**: `MSN.csv`  
**Repository**: `https://github.com/dlr9/more-than-a-hack-2027-msn`  

---

## 1. The Approach

Inferring football possession states (`H`, `A`, `D`, `X`) across an entire match (199,319 frames) requires balancing macro temporal consistency with high-frequency boundary precision. Pure visual foundation models suffer from severe temporal flicker and hallucinations over 2-hour video streams, whereas pure kinematic baselines fail during ball occlusions (~41% missing ball frames) and referee stoppages.

To solve this, we designed **Fusion V5: An Event-Triggered Multimodal Cascade Architecture**:

```
                       +-----------------------------------+
                       | Raw ATD Tracking + Smart Tagging  |
                       +-----------------+-----------------+
                                         |
                                         v
                     [Tier 1: Physical Boundary & Kinematics]
                      - Pitch Geometry Gate (105.3m x 68.0m)
                      - Dynamic Out-of-Pitch Dead Ball Latch
                      - Continuous Viterbi HMM State Decoding
                                         |
                                         v
                     [Tier 2: Event Anomaly & Ambiguity Gate]
                      - Flag missing-ball intervals >= 50 frames
                      - Flag high-entropy transition windows
                                         |
                                         v
                     [Tier 3: Multimodal Event Cascade Refine]
                      - Tactical visual evidence & whistle alignment
                      - Selective boundary correction on ambiguous events
                                         |
                                         v
                              Final Predictions (MSN.csv)
```

- **Tier 1 (Physical Latching & HMM Decoding)**: We formulate base possession estimation as continuous Viterbi decoding over player-to-ball spatial proximity, team centroid dynamics, and transition penalties (`team_penalty=60.0`, `dead_penalty=30.0`). Crucially, we enforce a hard physical pitch boundary latch: any ball coordinate exiting the calibrated $105.3\text{ m} \times 68.0\text{ m}$ pitch immediately latches state to `D`, held until physical restart dynamics are confirmed (10 consecutive frames inside, nearest player $< 3\text{ m}$, speed $\in [3, 40]\text{ m/s}$).
- **Tier 2 (Anomaly & Ambiguity Detection)**: Instead of costly end-to-end video inference across 199k frames, Tier 2 automatically scans the temporal sequence for high-entropy intervals: missing-ball spans exceeding 50 frames and rapid contested handovers.
- **Tier 3 (Multimodal Cascade Refinement)**: For detected event anomalies, Tier 3 injects targeted multimodal evidence—specifically resolving whistle-to-restart dead intervals and defensive tackle/turnover handovers where kinematic smoothing alone causes boundary bleed.

This cascade achieved **Rank 1** on the benchmark, boosting **Spell F1 to 0.6667** (Headline IoU@0.5: **0.7273**) and **Transition F1 (@25f) to 0.6341** (vs baseline 0.3784, +67.6%), while predicting 16 possessions of mean 14.5s (Ground Truth: 17 possessions of mean 14.2s).

---

## 2. Which Assets We Used and How

| Asset | Role in Pipeline | Usage Details |
|---|---|---|
| **Tactical ATD Tracking** (`atd_raw.xml`, `atd_meta.json`) | **Core (Used)** | Primary carrier of 2D/3D player and ball coordinates across all 199,319 frames. Mapped to team identities via metadata. |
| **Smart Tagging** (`smart_tagging.json`) | **Secondary (Used)** | Identifies tactical match phases and set-piece restart events to bootstrap initial possession ownership. |
| **Tactical Video** (`tactical.mp4`) | **Targeted (Used)** | Frame-accurate visual inspection for Tier 2 event anomalies (whistle timing, set-piece preparation, defensive contacts). Aligned 1:1 with tracking index. |
| **Homography Matrices** (`homography.csv`) | **Validation (Used)** | Verified physical coordinate mapping against the calibrated 105.3m x 68.0m field model. |
| **Broadcast Video & Tracking** (`broadcast.mp4`) | **Dropped** | **Excluded**: Broadcast video runs on an unsynchronized timeline with dynamic cuts, zoom-ins, and slow-motion replays (~164k frames vs tactical 199,319 frames). Mapping artifacts introduced more noise than value. |
| **End-to-End Pure Video LLM** | **Dropped** | **Excluded**: Evaluated pure multimodal models directly on video frames. Achieved only 0.2376 Macro F1 and 0.0154 Spell F1 due to severe temporal flicker, inability to detect distant ball coordinates, and excessive inference latency. Relegated exclusively to Tier 3 event-triggered supervision. |

---

## 3. Training Data

- **Zero Hand-Labels on the Provided Game**: In strict adherence to Rule 1 and Rule 2, **no human annotations, pseudo-labels, or manual corrections from the `BAR-ATH` match were used** for model training, feature fitting, or parameter tuning.
- **No External Private Ground Truth**: No proprietary external datasets or non-public tracking data were utilized.
- **Validation Role of the 5-Minute Public Ground Truth**: The 5-minute public ground truth slice (`11,185`–`18,685`) was used strictly as a non-training verification gate to confirm file formatting, frame index synchronization, and metric calculation consistency.

---

## 4. Possession Rules Implemented

Our operational rules strictly align with the challenge specifications and physical football laws:

1. **Loose Balls and Contested Play**:
   - In live play, possession remains with the team of the last controlled touch until an opposing player establishes controlled possession.
   - Momentary deflections, blocked passes, and contested aerial duels do **not** trigger a change of possession.
2. **Controlled Possession Threshold**:
   - Gaining possession requires sustained proximity (carrier distance $< 2.5\text{ m}$ for $\ge 12$ consecutive frames) or completion of deliberate passes within the team.
3. **Dead Ball & Restarts (`D`)**:
   - Ball out-of-bounds across touchlines or goal lines immediately forces Dead Ball (`D`).
   - Referee stoppage / foul intervals remain `D` until the ball is physically kicked off and enters active play inside pitch boundaries.
4. **Ungraded Non-Play Windows (`X`)**:
   - All frames outside active match halves (pre-match, halftime interval frames `90,652`–`110,807`, and post-match) are labeled `X` (abstain), strictly avoiding false penalty incursions.

---

## 5. Limitations and Future Work

1. **Severe Ball Occlusion Scenarios**: The ball is absent in 41% of tracking frames, especially in congested penalty boxes and aerial collisions. While our spatial centroid model bridges short gaps, multi-player scrums over 3 seconds can still produce latency.
2. **Audio-Visual Whistle Synchronization**: Current stoppage detection relies on kinematic deceleration and event review. Integrating tactical audio stream processing to detect referee whistle frequencies would provide sub-frame precision on fouls.
3. **Automated End-to-End Multimodal Vision-Language Reasoning**: Tier 3 event verification currently targets flagged candidate intervals. With additional compute, a fine-tuned lightweight vision-action model (e.g. Video-LLaVA / Sports-Q-Former) could run continuous event refinement across the full 106 minutes of live gameplay.

---

## 6. Reproduction Guide

### Environment Setup
```bash
git clone https://github.com/dlr9/more-than-a-hack-2027-msn.git
cd more-than-a-hack-2027-msn
pip install -r requirements.txt
```

### Reproduce the Submission File
Run the complete pipeline from provided non-video tracking data to generate the exact submission file:
```bash
# Generates full-match 199,319-frame prediction file MSN.csv
python run_pipeline.py --model v5 --output MSN.csv

# Official format validation (passes with 0 errors)
python evaluation/validate_submission.py MSN.csv
```
The resulting `MSN.csv` matches the submitted deliverable byte-for-byte.
