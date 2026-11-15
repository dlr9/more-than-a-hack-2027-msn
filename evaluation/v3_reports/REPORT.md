# V3 optimization report

Public scores are development results, not independent full-match validation. All 120 candidates were fixed before search; every candidate uses continuous half-match decoding.

| Model | spellF1 | macroF1 | Dead recall |
|---|---:|---:|---:|
| V2 | 0.5935 | 0.7261 | 0.3996 |
| V3 | 0.4923 | 0.8192 | 0.7025 |

Promotion eligible: False. Search never changes the default unless --promote is explicitly supplied and the gate passes.

V3 spellF1 bootstrap 5–95%: 0.2769–0.6868.

| Held-out block | spellF1 | macroF1 |
|---|---:|---:|
| 1 | 0.4727 | 0.7216 |
| 2 | 0.3273 | 0.8639 |
| 3 | 0.6857 | 0.7189 |

Each block selects parameters on the other two blocks; fitting excludes 125 boundary frames. Historical label exposure and model design choices prevent interpreting these as independent test results.

| Ablation | spellF1 | macroF1 |
|---|---:|---:|
| no_confidence | 0.4462 | 0.7722 |
| no_visible_dead | 0.4923 | 0.7179 |
| no_dynamic | 0.4923 | 0.8192 |
| no_tags | 0.4167 | 0.8192 |
| no_repair | 0.4923 | 0.8192 |

Full-match diagnostics: [{"dead_fraction": 0.46279587753407075, "switches": 120, "median_segment_seconds": 19.32}, {"dead_fraction": 0.5069530167527396, "switches": 117, "median_segment_seconds": 18.14}]

No video, private labels, or manual prediction corrections were used. See summary.json for diagnostics and errors.json for error intervals. Search records and fingerprints remain local generated artifacts.