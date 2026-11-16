# R12 status: NO-GO for promotion; keep reproducible research tooling

Full raw-to-score experiment completed. New fits: 0. Frozen model heads reused. No default inference model changed. All three matches are previously consumed development matches; these are legal external source evaluations, NOT independent blind validation.

| Match/domain | Reference smooth50 | Frozen PR1 V3 | M3 | HC_M3 | Five-state global | Δ vs HC |
|---|---:|---:|---:|---:|---:|---:|
| J03WPY/natural | 0.317085 | 0.348423 | 0.557107 | 0.573158 | 0.535474 | -0.037683 |
| J03WPY/long | 0.261006 | 0.287189 | 0.475184 | 0.480720 | 0.449721 | -0.030999 |
| J03WOH/natural | 0.301146 | 0.351246 | 0.551485 | 0.584968 | 0.571601 | -0.013367 |
| J03WOH/long | 0.265022 | 0.302986 | 0.489915 | 0.506765 | 0.492529 | -0.014236 |
| J03WQQ/natural | 0.360843 | 0.370954 | 0.585319 | 0.596818 | 0.601709 | +0.004892 |
| J03WQQ/long | 0.311811 | 0.305586 | 0.480095 | 0.489049 | 0.490164 | +0.001115 |

Natural mean: 0.569595 vs HC_M3 0.584981 (−0.015386). Long synthetic occlusion: 0.477471 vs 0.492178 (−0.014707). Natural IoU 0.5 gained 30 previously unmatched GT spells but lost 138 matched spells, net −108. Even the third-match F1 increase hides a net loss of 35 true matches because fewer predictions reduce the denominator. Short <1s matched spells fell from 18/140 to 8/140 at IoU 0.5. Only one natural match improved F1, far below the frozen ≥2/3 and mean +0.02 gate.

## Mechanism and counterexamples

The fixed switch penalty suppresses genuine acquisition when geometry does not generate an anchor. Correct anchors cover only 8/57, 7/46, 7/37 subsecond GT spells; correct anchor starts appear within 1s of only 109/324, 110/301, 115/302 true direct handovers. High frame precision mostly describes easier sustained control, not transition recall.

The five-state topology also admits a cheap H→D→A route: its two transition costs total 0.4 versus 3 for unsupported direct acquisition. First-match wholly false opposite-owner Dead bridges increase 2→65. This is an architectural counterexample, not evidence for tuning another penalty. Independent Dead evidence must govern admissibility, and score-aligned training must account for short-spell recall. No post-score parameters were changed.

Team-shuffled anchors beat real anchors on WOH natural and WQQ long. Mean long real performance is below shuffled control. The evidence is insufficient for claiming information gain. Dead frame accuracy alone hides many brief cuts; full boundary counts, gained/lost GT matches, same-team restarts, and stratified errors are in R12_RESULTS.json/R12_VERIFICATION.json/R12_DEAD_BRIDGE_DIAGNOSTIC.json.

## New source finding

The CC BY4 ELASTIC human-timed public J03WPY event file was actually downloaded, blob-verified, and source-frame aligned. Of 265 cross-team receipts, only 15 coincide with an anchor and 73 have a same-team anchor within 1s. At-time source H/A agrees for only 124/265: receipt/contact semantics and source possession labels require explicit adjudication, not automatic relabeling. This is additional supervision for a future dense all-timeline action model, not an inference oracle or demonstrated spell gain.

## Scope and limitations

- Sparse global five-state DP uses soft transition penalties, not a learned duration semi-Markov objective. Future observations participate in backtracking, but a wrong anchor can still influence a long span when competing evidence is weak.
- Control anchors reuse old physical geometry; provenance explicitly marks independence=false. No new visual sensor signal was tested. Visual and Smart Tagging removal are N/A because neither was an input.
- Natural DFL inputs have zero missing-ball frames. Long is label-independent synthetic stress, not proof about natural camera occlusion. Dead labels may still exploit the historical DFL coordinate-freezing shortcut.
- PR1 comparison is the first preselected frozen V3 SkillCorner fold, not an ensemble. V2_idea_untuned is an illustrative generic baseline, not the teammate tuned implementation.
- No sealed WOY/WR9 access, no official five-minute GT, no target observations, no target-derived outputs. All historical models, failures, and caches retained.

## Decision and next evidence

NO-GO / REVERT candidate promotion. KEEP source, tests, full predictions, diagnostics and hashes. Roll back operationally to unchanged HC_M3/M3; no checkout or artifact deletion needed. Do not repeat geometry/smoothing searches. The next justified experiment needs genuinely independent event observations or human control-acquisition supervision with non-event negatives, tested on untouched complete matches. Resolve receipt-vs-control semantics and new data access before training. The present source audit does not authorize SoccerTrack terms or establish visual-model overlap safety.

## A/B/C ownership

- A (authorized local submission owner): confirm exact deadline/timezone, Form receipt and required documents. Current status UNKNOWN; user previously thought not submitted. Do not wait for this failed candidate.
- B (authorized isolated runner owner): validate existing Tactical chain, required assets, frame/label CSV schema and official offline run. TARGET_ASSETS/RUNNER_VALIDATED/CSV_VALIDATED remain UNKNOWN. Send status only, never target files or derived results.
- C (research/public release owner): review Draft PR, source licenses, full-match metrics and failure mechanisms. Retain model caches. Obtain a legal paired full-match action/trajectory dataset and enough isolated compute before visual inference.

## Recovery

Worktree: `/Users/dailuren/Documents/ChatGPT/barca/r12`; branch `codex/r12-control-global-20261116`; base a0acdbaa3b69186f3e161e58a288b2f96fea0693. Protocol/source pre-score commit `0999cf7`; every row records its exact full SHA and model/data/code/protocol hashes.

```sh
cd /Users/dailuren/Documents/ChatGPT/barca/r12
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2
.venv/bin/python -m unittest discover -s tests
# Preserve existing output files. Run these commands in a fresh result directory/copy for replay:
.venv/bin/python -m scripts.research_r12 run
.venv/bin/python -m scripts.verify_r12
.venv/bin/python -m scripts.diagnose_r12_bridges
PYTHONPATH=local_artifacts/r12/parquet-runtime .venv/bin/python -m scripts.audit_r12_external_events
```

Writers intentionally refuse overwriting prior results. Existing six raw-evaluation rows are resumable; aggregate writers remain append-only. `local_artifacts/r12/global-v1/*.npz` holds all full predictions, latent states and source anchor intervals; ignored and not uploaded. Public JSON contains aggregate metrics and SHA256 receipts only. Isolated optional parquet runtime: duckdb==1.4.1. Public event download URL and exact blob/hash are in R12_EXTERNAL_EVENT_AUDIT.json.

Tests: 225 local + 225 clean-source tests passed; 24 exact decoder/latent replays; 54 full official-vs-ROI score equivalences. Raw 44-feature reconstruction matched cached inputs, and HC_M3 regeneration exactly matched frozen predictions for all six match/domain runs. Publication and clean export verification receipts are recorded separately.
