# B → R9 contract (source-only port)

Source branch `feat/data`, commit `a54219dbff8948507b112234127816c133f7c08b`.
License: repository MIT, copyright 2026–2027 Metrica Sports; preserved in root `LICENSE`.
The MIT scope explicitly excludes match assets. No match files/configuration/report was read or imported.

| Source signature / behavior | New interface | Deliberate change / test |
|---|---|---|
| `fusion_v2.features(data_dir, cache=None)`; slot team 0/1 and speed /3.6 | `adapt_tracking(...)` | Already parsed arrays only, explicit bijection H=0/A=1 and units, no loaders or source discovery |
| B normalized geometry scaled by global pitch | `coordinate_units`, `pitch_m` required | Centered metres; no match dimensions or half constants copied |
| B static slot assignment | `player_velocity(batch)` | Identity-linked positions with half/gap reset; moving slot, renamed identity regression |
| B `features` Team attack / `defending in*` support | `tactical_evidence(...)` | Defense restricted to the three documented categories from `possession_detector.build_advanced_model`; no prefix guesses; explicitly unknown teams |
| B phase-wide floating tags | `soft_team_support(...)` | Uncalibrated soft evidence only; max avoids duplicate amplification; contradictory overlapping teams abstain |
| B set-piece tag supports D; V1 first half interval hard D | existing R9 `restart_proposals(...)` | Only bounded restart candidate; no whole-interval D assignment |
| B `predict(f, cfg, blocks=None)` + runner auto-selected configuration | `predict_only(..., baseline_decoder=..., tag_logit_weight=0)` | Explicit supplied decoder; no built-in model selection, config reads, scoring, subprocess, or saving |
| B fixed 25Hz and source kickoff constants | Explicit fps/frame/period arguments | 10/25Hz and half/gap tests; no target anchor copied |

`predict_only` zero weight or absent tags passes identical probability heads into the same frozen decoder.
No claim that the supplied decoder is automatically M3: caller must explicitly supply and pin it.
Nonzero tag support changes conditional Home log odds only; Alive inputs remain unchanged.
Joint decoding can still change D outputs indirectly, so performance must be assessed in a separately frozen experiment.
No real-data tag performance experiment was run by this integration module.
Missing ball/player coordinates remain NaN. An unknown or absent observation never implies D or zero motion.
Returned reported speed is a separately named observation, not automatically a legal model feature.
Source truth/phase fields have no place in this array API.

The prior default/config values and all search utilities are intentionally absent.
Independent external calibration would require new provenance and evaluation; these tests establish contracts only.
