# Possession Fusion V2

From this repository directory:

```powershell
python models/optimize_v2.py --data-dir F:\non_video_dataset --trials 60
python run_pipeline.py --model auto --skip-extract
python -m unittest discover -s models -p test_fusion_v2.py
```

Inference accepts `--model v1|v2|auto`, `--config <json>`, `--data-dir` and
`--output <csv>`. Auto reads `models/default_model.json`, written by the release
gate. V2 verifies SHA-256 source hashes and cache version on every invocation;
`--skip-extract` never bypasses this check. V1's unversioned cache is regenerated
when invoking the pipeline. Existing V1 prediction files remain the comparison
baseline during optimization.

Only NumPy and pandas, already required by the project, are needed. Search is
CPU-only, seed 20261115, with at most 100 configurations per decoder. Default is
60 each. No labels enter the feature extractor or inference module. XML speed
units are km/h and are converted to m/s; normalized coordinates become metres.
ATD frame IDs are one-based video frames, distinct from the IdFrame clock.

The feature cache stores player coordinates, observed ball positions, valid
player speed summaries and tactical phases. Short ball gaps are linearly
reconstructed within each half, with reduced confidence. Long gaps remain
unknown. Tactical tags are soft evidence, not ground truth. Viterbi optimizes
the sum of observation scores minus switching penalties independently per half;
the alternate state decoder requires sustained evidence and backdates confirmed
changes. Both are offline algorithms and may use future observations.

Three contiguous public windows are used for blocked selection. Each held-out
window selects parameters using the other two windows, excluding 125 frames
near internal boundaries from fitting. Final parameters are selected on all
three fitting windows; their full public score is a development score, not a
held-out test score. Historical V1 tuning has already exposed these labels.
Bootstrap intervals measure sampling variation, not tuning bias or full-match
generalization. No private full-match truth is available.

`evaluation/v2_reports/experiments.json` contains all configurations, per-class
metrics, confusion matrices, transition errors, interval IoU metrics, bootstrap
intervals, fold-selected results, ablations and full-match sanity summaries.
`REPORT.md` provides a compact comparison. V2 replaces the default only if its
continuous inference improves both full public and mean window spellF1 without
reducing full public macroF1. Otherwise V1 stays the default and the candidate
CSV remains available for inspection.
