# V2 development experiment protocol — 2026-11-14

This experiment is development evidence, not a new final test. It reuses PR #1's 44-feature design and two HistGradientBoosting classifiers, with a fixed Viterbi baseline. No target match inputs, public target GT, or private reports are permitted.

- Source: SkillCorner Open Data commit `4340d274572876239c154c90bc507a9b3250a656`; Metrica sample-data `e706dd506b360d69d9d123d5b8026e7294b13996`. Exact retrieved checksums are in the external manifest.
- SkillCorner IDs: 1874553, 1886347, 1899585, 1927964, all from the previously used training pool. Original test IDs 1996435, 2011166, 2015213 stay sealed.
- Four deterministic folds: held-out match at index i, calibration match at (i+1)%4, remaining two matches train. Each fold uses disjoint matches, but all four were used historically; results are NOT untouched holdout evidence.
- Fixed model: 60 iterations, 15 leaves, learning rate .1, L2 1, seed 2027, no random frame early-stopping split. Training sampled at 2 Hz, evaluation uses complete 10 Hz timelines. No hyperparameter search or result-based rerun.
- Three observation variants: all available players; detected players only; detected players with additional input-level contiguous ball dropout (.5/2/5 seconds, every 20 seconds with deterministic jitter). Original and augmented training samples are concatenated. All dependent spatial/speed/temporal features are rebuilt before model fitting.
- Conservative labels: phase presence provides weak live supervision, phase absence stays unknown. This does **not** supply a reliable dead class, so the conservative alive classifier must fail closed. An explicitly labeled legacy phase-complement alive head is fitted only for diagnostic ablations; it is never promoted.
- HA labels retain stable possession and phase agreement. Synthetic dropout retains the source HA label; this tests missing-input robustness, not newly observed natural missing-ball truth.
- Binary logistic calibration uses the separate calibration match without class reweighting. Report before/after Brier, log loss, ten-bin ECE/reliability, plus HA macro F1 and missing-ball strata.
- Fixed decoder costs: H/A 3, D 1. Experimental control proxy requires observed ball, nearest distance <1.5m, opponent margin >2m, and .3 seconds of same-team evidence within a period. It lowers only the supported H/A transition cost. It is proximity evidence, not a verified interception/tackle/restart detector.
- Decoder boundaries use period metadata only. Unknown label intervals remain in inference. SkillCorner HAD numbers use weak legacy phase-complement labels; Metrica HAD numbers use partial event interiors. Neither qualifies for a full-spell promotion gate.
- Metrica is one repeatedly evaluated external game, not four independent observations; no confidence interval treating folds as independent Metrica matches. SkillCorner intervals resample four held-out matches, 2,000 replicates, seed 2027; n=4 is weak evidence.
- Promotion: disabled regardless of scores until independent complete HAD timelines, calibration coverage, and a fresh holdout are available. Fallback remains the official reference and PR #1's isolated safe snapshot.

Commands:

```sh
python scripts/fetch_external_v2.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/experiment_v2.py
python -m unittest discover -s tests -v
python scripts/check_public_tree.py
```

The fetcher retrieves only the explicit public IDs. Raw data, caches, model objects, and frame predictions remain in ignored `local_artifacts/`. The public reports contain aggregate other-match results only.

## R2 post-hoc diagnostic

After R1 showed cross-provider D collapse, a separate `--followup` run kept alive training identical to the observed variant and augmented only HA training. It reuses the same development folds and domains, so it is not a fresh independent test and cannot promote a candidate. Original R1 outputs remain intact. Results: [follow-up aggregates](external_followup_v2.json).
