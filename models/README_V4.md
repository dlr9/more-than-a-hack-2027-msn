# V4 conservative refinement

V4 freezes V2 H/A emissions, including tactical-tag smoothing. Only D receives a
nonnegative boost supported by visibility, slow-player fraction, low mean player
speed, and either low ball speed or low local ball position range. Zero boost
with V2 penalties reproduces V2 exactly.

The base study used 42 fixed configurations. A declared second stage added 21
configurations with team penalties 45/50/55 and dead penalty 30. Both stages are
development on repeatedly used public labels, not independent tests. The original
base study is retained separately; the second stage re-evaluates all 63 candidates.

Selection pools official F1 over retained training frames instead of averaging F1
on separately clipped training blocks. Adjacent retained blocks stay connected.
Each held-out selection excludes the held block and a 125-frame buffer. Final
refit uses complete public labels, explicitly as a development score. Promotion
requires higher full and mean-block spellF1, no decrease in full macroF1, and
held-out mean spellF1 at least as high as V2 on the same windows.

From the repository directory:

```powershell
python models/optimize_v4.py --refine
python models/optimize_v4.py --refine --reuse-search --promote
python run_pipeline.py --model v4 --config evaluation/v4_refine_reports/candidate_config.json --output evaluation/results/model_fusion_v4_candidate.csv
python -m unittest discover -s models -p 'test_fusion_v*.py'
```

Search writes candidate_config.json and candidate.csv inside its study directory.
Only accepted promotion writes the deployed model config, prediction and default
manifest. Rejected or ordinary searches cannot overwrite a deployed V4. The
default remains V2 for this study because primary spellF1 did not increase.

Reuse fingerprints cover raw data, public labels, model/search/scoring code,
metadata and tag-loader helpers, frozen V2 config, actual feature-cache bytes,
windows and candidate list. Changed dependencies invalidate reuse. Feature
extraction remains unchanged V2/V3 code with versioned caches; any future change
to extraction logic must invalidate/rebuild those caches too.

No video, external dataset, private labels or hardcoded truth intervals are used.
Official scoring and frozen V2 config remain unchanged. Existing notebook work is
outside this change. Search records, caches and prediction CSVs remain local.
