# R6: train external H/A/D models and improve missing-ball decoding

Proposed base: `codex/pr1-v2-integration-20261114` (stacked on Draft PR #2).
Proposed head: `codex/r6-train-idsse-20261114`. Create as Draft after team publication approval.

The existing SkillCorner labels could not support complete H/A/D spell evaluation. This change adds verified public IDSSE acquisition, streaming and independent parser checks, a schema3 feature contract, and real two-head HGB training with three-match leave-one-out evaluation. It preserves the existing 44-feature interface and frozen V3 artifacts.

Across three public external matches, full source-label spell F1 averages 0.3569 for the fixed V3 adapter, 0.5610 for M1 and 0.5646 for M2. Raw-input dropout augmentation improves stress spell F1 from 0.4071 to 0.4361; the preregistered missing-observation decoder improves it to 0.4611 with unchanged natural predictions. M1 is retained as a research baseline; M2/M3 remain experimental. F1 reweighting was reverted; weak Metrica Alive training is reported separately. Seventeen new HGB classifiers were actually fitted.

Validation: 78 synthetic/regression tests pass, including a clean source copy without data caches; six halves agree with floodlight on labels, transitions and ball coordinates; nine saved IDSSE bundles replay both domains exactly. Twelve frozen V3 files retain their hashes. New remote CI has not run before publication.

The three development matches share a league and home team, and DFL dead-ball positions contain a strong freezing cue. These results establish external development performance only. The sealed matches remain unused; no target assets, public target GT or competition outputs were processed. The new Mac reference/preflight entry points have synthetic tests only and require authorized, isolated human offline operation.

Reproduction and rollback: `docs/R6_REPRODUCE.md`, `docs/R6_MODEL_REGISTRY.json`, `docs/R6_HANDOFF.md`. Models and raw XML remain outside Git; public changes contain code and aggregate results only. No main merge or competition submission is requested by this PR.
