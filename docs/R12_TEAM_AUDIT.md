# R12 integration audit — public source only

Inspected remotely at `feat/data` a54219dbff8948507b112234127816c133f7c08b. No target input, target prediction, selected configuration, or private report was opened. No unrelated-history merge was attempted.

| Source | Git blob | Finding |
|---|---|---|
| models/fusion_v2.py | 6a49720993fd9c015e05f45d148480567c04bd25 | A set-piece interval contributes Dead evidence over its entire span. Missing ball contributes Dead evidence. Proximity and phase votes do not establish controlled acquisition. Sustained-state thresholds and fixed switch costs can remove short spells. |
| models/fusion_v3.py | bfd98345685742e86219cb58676cef654f50bc24 | Adds visibility/coverage and player motion, with different enter/leave Dead costs. Still inherits interval-wide set-piece and missing-ball Dead assumptions. |
| models/fusion_v4.py | b21f587e73d0362ec649c5a692b292e3dc1c1225 | Motion/stationarity Dead boosts build on a fixed V2 configuration; not an independent control observation. |

All three retain dependencies on target game blocks and fixed match-specific restart handling. These constants and GT-selected coefficients are quarantined. We did not execute these modules. Generic visibility contracts and separate state-entry/state-exit logic are portable ideas, not validated parameter sets. `V2_idea_untuned` is a newly defined distance-only Home baseline plus frozen Alive; it is explicitly NOT a performance reproduction of teammate V2.

R11.5 `r11p5/timeline.py` assigns `out[i:b]=team`: one accepted erroneous proposal can change the rest of an original baseline spell, however long it is. Baseline boundaries terminate the overwrite; subsequent observations are not jointly optimizing that suffix. R12 uses full-half backtracking, preserving a counterexample test where later strong evidence defeats an isolated false anchor. This repairs the hard-write structure, but does not establish empirical superiority.

Control-episode geometry is inherited unchanged, with provenance attached. It is NOT independent of the original geometric model. Outside extracted intervals the signal is UNKNOWN; no contact, phase, or nearest-player event alone is converted into a control acquisition. Returned confidence is an uncalibrated evidence grade, not a probability.
