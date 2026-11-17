# R13: real training completed; no candidate promoted

Three new fitted models, 72 full-match variant/domain scores, 72 exact prediction replays and 72 official/ROI score equivalences. 237 local and 237 clean-source tests pass. Official scoring source is unchanged. Research code is retained; none of these candidates replaces the frozen model.

## Results and decisions

Mean official spellF1 across IoU 0.3/0.4/0.5/0.6/0.7:

| Match/domain | HC_M3 | Memory topology | Dense human events | Prior + neutral pool (primary) |
|---|---:|---:|---:|---:|
| WPY/natural | .573158 | .481571 | training match | training match |
| WPY/long | .480720 | .408499 | training match | training match |
| WOH/natural | .584968 | .495597 | .538556 | .498475 |
| WQQ/natural | .596818 | .520654 | .569021 | .531304 |
| WOH/long | .506765 | .441071 | .477388 | .444156 |
| WQQ/long | .489049 | .445251 | .465929 | .428664 |

All matches are legally external but previously consumed development matches: NOT untouched independent validation. Natural DFL has no missing-ball frames; long is synthetic label-independent occlusion, not natural occlusion validation. No hidden competition score or ranking claim.

1. Memory experiment (pre-score 5b01241): splitting Dead by previous owner prevents a cheap H→D→A shortcut. Natural false opposite-owner D bridges fall from 65/55/83 to zero. Yet spellF1 falls on every match: absent acquisition anchors now suppress real handovers. The mechanism repair is real; the model is NO_GO.
2. Dense supervision (pre-fit 203f8a5): three Logistic models actually trained on public WPY (human control-event timing, source transitions, shuffled event teams). Forty-eight identity-coherent motion features at three time offsets, mirrored team augmentation, dense non-events and synthetic missing-ball views. Predict every frame; no evaluation event times are inputs. Human model fails against HC and negative controls on WOH/WQQ. Event semantics and persistent ownership differ. NO_GO.
3. Probability factorial (pre-score 1e505c3): balanced class outputs do not represent population priors; arithmetic mixing can alter the baseline even for neutral evidence. Training-prior correction plus a neutral-preserving log-odds pool fixes those specific semantics but worsens primary spellF1. The prior-only WQQ increase is not grounds to choose that arm post hoc. NO_GO.

Dense/calibration keep the exact HC Dead mask, hence do not solve Dead detection. Their H/A boundary labels can still alter team-specific restart scores. Full metrics include five IoUs, macro F1, gain/loss of true matches, subsecond/1–5s/long recall, D errors, and handover/restart errors. Memory and dense rows also include visibility strata and false D cuts. See the three RESULTS JSONs; controls are retained, not selected away. Visual/Smart Tagging ablations are N/A because neither is used.

## Why change direction

Post-hoc R13_EVENT_DIAGNOSTIC.json records ownership agreement on event-dominant GT-alive frames: human expert 74.6%/74.7% versus HC 86.3%/85.0% on WOH/WQQ. Even on training WPY the expert is only 77.0%. This is a diagnostic of this representation and target, not proof that all learned event models fail. Receipt/contact labels cannot silently stand for persistent controlled possession. Repeating geometry penalties, smoothing, or the same model with different names is unsupported.

The next justified route requires actual independent action evidence (public visual control/contact or referee audio), or a properly adjudicated full-match control-acquisition dataset with non-events and ambiguity. Existing OSL/PathCRF audits have unresolved legal/input/compute/data-pairing gates; no new visual inference is claimed. Do not use target assets to fill the gap. R12 has comparisons to reference algorithm, frozen PR1 V3, M3 and generic untuned V2 idea; this increment compares against the stronger unchanged HC and controls rather than rerunning unchanged baselines.

## Teammate update and submission

feat/data advanced to ff84c79fc8bad6742e22e6bd975a5eef36f4d3e7. Allowlisted V5 source unexpectedly embeds target-specific label overrides. Review stopped; no such values are copied into this branch, executed, or used for research. Output/report files were not opened. See R13_TEAM_UPDATE_REVIEW.md. Do not adopt V5 as a general multimodal result without a clean, isolated implementation and external validation.

Official public submission instructions say November 17, 2026; the exact time/timezone is not stated. Form URL is emailed to registered participants. A scoped Gmail search was attempted but the connector returned Unknown tool, so registration, receipt and final submission remain UNKNOWN. See R13_SUBMISSION_HANDOFF.md. This research delivery is not a competition submission.

## A/B/C TODO and rollback

- A: Authorized submission owner must recover the registration Form and confirm deadline, registered team name and receipt; no receipt exists in this session.
- B: Isolated local runner owner must remove target-specific output overrides, confirm generic inference, assets and CSV validation, and pair the exact generating revision with the technical document. No target files or derived outputs should enter cloud research.
- C: Research/release owner reviews this Draft PR and keeps HC_M3/M3 as fallback. Obtain clean independent action evidence and adequate disk/compute before a new method. Do not consume sealed WOY/WR9 or pretend already observed matches are blind.

Operational rollback requires no file deletion or checkout: leave existing default inference unchanged and use the frozen HC_M3/M3 path. All model binaries, probabilities, historical results and failed candidates remain in ignored local cache. KEEP code/tests/diagnostics; REVERT candidate promotion. New model/score hashes are in R13_MODEL_REGISTRY.json and each result row.

## Exact recovery

Worktree `/Users/dailuren/Documents/ChatGPT/barca/r13`; branch `codex/r13-control-recovery-20261116`; base e22a2a5fbc847b2d14fce55e8593eb91d5a7db0c. Protocols record immutable inputs/code hashes. Result rows record exact scoring SHAs. The final delivery SHA is the Git commit containing this document and is verified in the publication receipt; it is not embedded recursively in itself.

```sh
cd /Users/dailuren/Documents/ChatGPT/barca/r13
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2
.venv/bin/python -m unittest discover -s tests
# Use a NEW output path on each replay; never overwrite old receipts.
.venv/bin/python -m scripts.verify_r13 --out /tmp/r13-verification-new.json
.venv/bin/python -m scripts.diagnose_r13_events --out /tmp/r13-diagnostic-new.json
```

Full original execution commands, only in a fresh reproduction copy whose R13 result paths do not yet exist (keep all upstream inputs):

```sh
.venv/bin/python -m scripts.research_r13 run
PYTHONPATH=local_artifacts/r12/parquet-runtime .venv/bin/python -m scripts.train_r13_dense run
.venv/bin/python -m scripts.evaluate_r13_calibration run
```

Do not refreeze protocols or rerun fits in the existing cache. Writers intentionally refuse overwriting historical files. Python 3.12, scikit-learn 1.7.2; optional parquet reader duckdb 1.4.1 is isolated. Public-source event provenance and license are in inherited R12_EXTERNAL_EVENT_AUDIT.json. No model binary or raw/derived target data is published.
