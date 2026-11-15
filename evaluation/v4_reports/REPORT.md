# V4 conservative refinement

All public results are development scores. Forty-two fixed candidates; no additional search after observing results. V2 team emissions are unchanged. Zero boost with V2 penalties exactly reproduces V2.

| Model | spellF1 | macroF1 | Dead recall | Predicted spells |
|---|---:|---:|---:|---:|
| V2 | 0.5935 | 0.7261 | 0.3996 | 14 |
| V4 | 0.5935 | 0.7929 | 0.6413 | 14 |

Promotion eligible: False. Held-out mean spellF1 0.5624; V2 block mean 0.5624.

Selection uses pooled official F1 on the retained frames, not an average of cropped-spell F1s. Held-out labels and their 125-frame buffer are masked for each fold selection. Final refit uses all public labels, explicitly as development data.

| Held-out block | spellF1 | macroF1 |
|---|---:|---:|
| 1 | 0.5538 | 0.8109 |
| 2 | 0.4667 | 0.5563 |
| 3 | 0.6667 | 0.6823 |

| Ablation | spellF1 | macroF1 |
|---|---:|---:|
| no_visible_boost | 0.5935 | 0.7261 |
| v2_transition_penalties | 0.5935 | 0.7929 |

V4 spellF1 bootstrap 5–95%: 0.4000–0.7871.

Public labels have historical tuning and design exposure. Cross-validation and bootstrap do not establish independent full-match improvement.

Full-match diagnostics: [{"dead_fraction": 0.4662438496482817, "switches": 169, "median_segment_seconds": 11.1}, {"dead_fraction": 0.5002645169416803, "switches": 161, "median_segment_seconds": 10.74}]