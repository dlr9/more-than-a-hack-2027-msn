# R7B checkpoint — E01 complete, E02 implementation

- Branch: `codex/r7b-player-only-missing-alive`, continued from R7, never main.
- last_verified_sha: `e498e9390189bdf22ea3549ebf5f839b45915dbc` (R7 completed checkpoint).
- R7 remote objects verified through that SHA; no new ref/PR yet. PR #2 remains unchanged.
- Frozen 30 R6 + 9 R7 model hashes verified; inherited 88 tests passed.
- E01: 195 source-known H/A frames are 4 missing-ball episodes in SkillCorner 1874553. All 195 also have ZERO detected players. M2/M3 mean Alive .42721; argmax D=195 and decoded D=195. Exact frame/period/label/ball missing alignment passed. No independent live-motion evidence exists inside these 195 frames.
- Both J03WMX and J03WN1 were scored in completed R7. Neither is blind. Do not reuse as new model selection evidence.
- Preserve all `local_artifacts/r6`, `r7`, `v3` outputs read-only. New output root: `local_artifacts/r7b`.
- Current failure: synthetic slot-permutation test exposed cancellation in centered activity variance (~1e-7); replaced with centered-origin cumulative variance before any fit. No model change from results.
- OS process listing denied by sandbox; prior recorded run sessions completed, no new writer uses old experiment directories. New stages use exclusive output creation and resumable per-match files.
- Next exact command: `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -p test_player_only.py`
- Then run preregistered `scripts/research_r7b.py prepare`, `train`, `evaluate` after protocol/code checkpoint. No hyperparameter search.
- Owner A: independent feature contract, tests, recovery and Draft release. Owner B: model/episode errors, quality gate, counterfactual distinction. Owner C: authorized offline asset preflight and official deadline verification; Codex never opens target assets.
- Fallback: preserved R6-M1 research model; M2/M3 retain cross-provider failure warning. No model is target-validated or submission-ready.
