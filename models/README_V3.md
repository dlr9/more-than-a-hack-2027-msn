# Fusion V3

Run from the repository directory:

```powershell
python models/optimize_v3.py --data-dir F:\non_video_dataset
python models/optimize_v3.py --data-dir F:\non_video_dataset --reuse-search --promote
python run_pipeline.py --model auto --skip-extract
python -m unittest discover -s models -p 'test_fusion_v*.py'
```

Search uses 120 configurations generated before any scores are inspected, with seed
20261116. It never changes the default model. `--promote` only updates the default
if all three release conditions pass: full public spellF1 improves over frozen V2,
full public macroF1 does not decrease, and mean block spellF1 improves. If no
candidate passes, V2 remains the default. V3 remains available via `--model v3`.

All candidates and the V2 comparator perform continuous inference per half before
label masks are applied for scoring. Three fixed contiguous public blocks are used;
125 frames near internal boundaries are excluded from fitting scores. Parameters
for each held-out block are chosen using only the other blocks' fitting scores.
All public labels have historical exposure: these are development diagnostics,
not an independent estimate of private full-match performance.

V3 attenuates team advantage when both teams are far from the ball or tracking
coverage is poor. It combines player low-speed fraction, mean player speed and
1/2-second ball position ranges to detect visible-ball stoppages. Attack tags,
set-piece tags and missing observations have separate weights. Transitions into
and out of dead ball have separate penalties; dynamic variants reduce penalties
when stoppage or restart observations support the transition. Coordinate-derived
ball speed resets at half boundaries and rejects jumps above 40 m/s. Optional
short-gap repair requires plausible endpoint motion and reduced confidence.

The cache also retains player speed quartiles for diagnostics; current observation
scores use mean speed and slow-player fraction, not these quartiles. Speed in the
source metadata is km/h and is converted to m/s. All inference remains offline.

Reuse checks SHA-256 fingerprints covering source data, labels, relevant model and
search code, V2 config, scoring code, scoring windows, seed and the candidate list.
Changed inputs refuse reuse. The large search record and feature caches are local
artifacts; compact summaries, error intervals and the report are versioned.

V2 code and config remain unchanged. Existing notebook work is outside this change.
The canonical entrypoint is repository `run_pipeline.py`; the workspace-level
entrypoint delegates to it. Official evaluation code is unchanged.
