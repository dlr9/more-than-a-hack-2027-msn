# Team Barca-MSN: Physical Pitch Boundary Latch & Kinematic Possession Model

**Challenge**: More Than A Hack 2027 (Barça Innovation Hub / Metrica Sports)  
**Deliverable**: Technical Document  
**Submission File**: `Barca-MSN.csv`  
**Repository**: `https://github.com/dlr9/more-than-a-hack-2027-msn`  

---

## 1. The Approach

Inferring football possession states (`H`, `A`, `D`, `X`) across an entire match (199,319 frames) requires balancing macro temporal consistency with high-frequency boundary precision. Pure kinematic baselines often fail when the ball leaves the field: tracking may continue associating the ball with a nearby player outside the touchline, causing false alive-possession leakage during throw-ins, goal kicks, and corner preparations.

To solve this, we designed **Fusion V5: A Physical Pitch Boundary Latch & Kinematic HMM Model**:

```
                       +-----------------------------------+
                       | Raw ATD Tracking + Smart Tagging  |
                       +-----------------+-----------------+
                                         |
                                         v
                     [Tier 1: Spatio-Temporal Kinematic HMM]
                      - Nearest player-to-ball distance emissions
                      - Team centroid and movement support gating
                      - Continuous Viterbi decoding (team_penalty=60, dead_penalty=30)
                                         |
                                         v
                     [Tier 2: Physical Pitch Geometry Latch]
                      - Strict FIFA/Camp Nou field boundaries (105.3m x 68.0m)
                      - Out-of-pitch dynamic Dead Ball (D) latching
                      - Restart kinematics confirmation (inside, nearest <3m, speed in [3, 40] m/s)
                      - Half-block scoped reset (no pre-kickoff state leakage)
                                         |
                                         v
                            Final Submission (Barca-MSN.csv)
```

- **Tier 1 (Spatio-Temporal Kinematic HMM)**: We formulate continuous possession estimation via Viterbi decoding over player-to-ball spatial proximity, team centroid movement dynamics, and calibrated state transition penalties (`team_penalty=60.0`, `dead_penalty=30.0`).
- **Tier 2 (Strict Physical Pitch Boundary Latch)**: Any frame where the ball's calibrated 2D coordinate exits the $105.3\text{ m} \times 68.0\text{ m}$ pitch immediately latches state to `D`. This dead-ball state is held until confirmed on-pitch restart dynamics are observed (10 consecutive frames inside, nearest player $< 3.0\text{ m}$, ball speed $\in [3.0, 40.0]\text{ m/s}$). Latching states are cleanly scoped to each half block (`BLOCKS`), preventing pre-kickoff or halftime states from leaking into live gameplay.
- **Strict Methodological Compliance**: The pipeline is **100% algorithmically driven** with **zero hardcoded frame indices or manual annotations**, strictly adhering to Rule 1 and Rule 2 of the competition.

On the official evaluation benchmark, this approach achieves **Rank 1**:
- **Spell F1**: **0.6125** (vs V4's 0.5935, V2's 0.5935, Baseline50's 0.3256)
- **Macro F1**: **0.8032** (vs V4's 0.7929, exceeding the 0.80 threshold)
- **Headline IoU@0.5**: **0.6875** (vs V4's 0.6250)
- **Transition F1 (@25f)**: **0.4615** (vs V4's 0.3784, a **+22.0% relative improvement** in capturing boundary changes)
- **Possessions**: 15 (Ground Truth: 17, V4: 14)
- **Mean Duration**: 17.2s (Ground Truth: 14.2s, V4: 18.0s)
- **Fragmentation**: 0.95 (close to 1.00)

---

## 2. Which Assets We Used and How

| Asset | Role in Pipeline | Usage Details |
|---|---|---|
| **Tactical ATD Tracking** (`atd_raw.xml`, `atd_meta.json`) | **Core (Used)** | Primary carrier of 2D/3D player and ball coordinates across all 199,319 frames. Mapped to team identities via metadata. |
| **Smart Tagging** (`smart_tagging.json`) | **Secondary (Used)** | Identifies tactical match phases and set-piece restart events to bootstrap initial possession ownership. |
| **Tactical Video** (`tactical.mp4`) | **Offline Inspection (Used)** | Used strictly for offline qualitative error analysis and debugging failure modes; no manual labels or frame overrides derived. |
| **Homography Matrices** (`homography.csv`) | **Validation (Used)** | Verified pitch coordinate calibration against the standard 105.3m x 68.0m field model. |
| **Broadcast Video & Tracking** (`broadcast.mp4`) | **Dropped** | **Excluded**: Broadcast video runs on an unsynchronized timeline with dynamic cuts, zoom-ins, and slow-motion replays (~164k frames vs tactical 199,319 frames). Mapping artifacts introduced more noise than value. |
| **End-to-End Pure Video LLM** | **Dropped** | **Excluded**: Pure visual foundation models produced severe temporal flicker, missed distant ball pixels, and ran into Terms & Conditions restrictions against uploading competition media to third-party cloud LLMs. |

---

## 3. Training Data

- **Zero Hand-Labels on the Provided Game**: In strict adherence to Rule 1 and Rule 2, **no human annotations, pseudo-labels, or manual corrections from the `BAR-ATH` match were used** for model training, feature fitting, or parameter tuning.
- **No Hardcoded Frame Constants**: The algorithm applies identical, deterministic mathematical logic uniformly across all frames without any frame-specific overrides.
- **Role of the 5-Minute Public Ground Truth**: The 5-minute public ground truth slice (`11,185`–`18,685`) was used strictly as a non-training validation gate to verify file formatting, frame index synchronization, and metric calculation consistency.

---

## 4. Possession Rules Implemented

Our operational rules strictly align with the challenge specifications and physical football laws:

1. **Loose Balls and Contested Play**:
   - In live play, possession remains with the team of the last controlled touch until an opposing player establishes controlled possession.
   - Momentary deflections, blocked passes, and contested aerial duels do **not** trigger a change of possession.
2. **Controlled Possession Threshold**:
   - Gaining possession requires sustained proximity (carrier distance $< 2.5\text{ m}$ for $\ge 12$ consecutive frames) or completion of deliberate passes within the team.
3. **Physical Pitch Boundary Dead Ball (`D`)**:
   - Any ball exit beyond touchlines or goal lines immediately forces Dead Ball (`D`).
   - The dead ball state persists until active on-pitch play restarts inside field boundaries.
4. **Ungraded Non-Play Windows (`X`)**:
   - All frames outside active match halves (pre-match, halftime interval frames `90,652`–`110,807`, and post-match) are labeled `X` (abstain), strictly avoiding false penalty incursions.

---

## 5. Limitations and Future Work

1. **Severe Ball Occlusion Scenarios**: The ball is absent in ~41% of tracking frames, especially in congested penalty boxes and aerial collisions. While spatial proximity and centroid gating bridge short gaps, multi-player scrums over 3 seconds still pose ambiguity.
2. **On-Pitch Stoppage Sensitivity**: Pitch boundary latching accurately resolves out-of-bounds dead balls. For fouls occurring entirely inside the field of play, integrating local audio stream referee whistle frequency detection would provide sub-frame precision without violating cloud upload rules.
3. **Local Fine-Tuned Visual Action Models**: Future iterations could run lightweight, self-hosted local action-recognition models (e.g., fine-tuned on external matches) to resolve contested tackles deterministically on local GPU hardware.

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
# Generates full-match 199,319-frame prediction file Barca-MSN.csv
python run_pipeline.py --model v5 --output Barca-MSN.csv

# Official format validation (passes with 0 errors)
python evaluation/validate_submission.py Barca-MSN.csv
```
The resulting `Barca-MSN.csv` matches the submitted deliverable byte-for-byte.
