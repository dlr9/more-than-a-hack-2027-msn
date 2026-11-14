# Fix public possession pipeline parsing, label provenance and validation gates

This is a code-only successor to PR #1, preserving its `solution/` architecture and 44-feature two-head/Viterbi approach. It repairs silent ATD parsing errors, Metrica time-varying player-slot mapping, observed/extrapolated player mixing, phase-complement dead labels, and decoder boundaries derived from label coverage. Schema/view checks, whole-match calibration diagnostics, synthetic dropout and an experimental control proxy are included.

Target-derived reports from PR #1 are intentionally not carried into this clean branch; their original contents/history require human review. No target data, public target GT or private report was accessed. No competition predictions were generated.

Validation: original 10 tests passed; repaired suite has 32 passing tests; dependency check and diff whitespace check passed. Four historical SkillCorner development matches were cross-fitted with separate calibration matches and tested on public Metrica Game 3 partial labels. R1 has 72 result groups; R2 has 24 post-hoc diagnostic groups. Both-head dropout caused severe D failure on Metrica; HA-only augmentation restored D F1 to approximately 0.887 but does not establish an improvement over the reference. Calibration/control-proxy gains are not consistently supported.

No model is promoted. The full independent H/A/D sequence gate remains untested; final output mode is disabled. Detailed hypotheses, file changes, metrics, failures, decisions, rollback and next steps are in `docs/engineering_review_v2.md`. This PR must remain draft pending team review; do not merge or submit competition files automatically.

Publication status at handoff: GitHub integration returned HTTP 403 `Resource not accessible by integration` for creating a tree. No remote repair branch/commit/PR was created; remote CI has not run. The local code branch and delivery archive are ready for publication when the connection has Contents and Pull requests write access.
