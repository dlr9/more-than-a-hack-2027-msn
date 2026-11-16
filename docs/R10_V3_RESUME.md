# R10 V3 recovery

Code worktree `/Users/dailuren/Documents/ChatGPT/barca/r10-v3`; branch `codex/r10-v3-team-integrated-20261116`; exact source base R9 PR5 `9f701a4f193d776515a6c2129d5809b6e736b8b2`. Old worktree `/Users/dailuren/Documents/ChatGPT/barca/review-pr1` remains on R9, unchanged. New HEAD and remote publication/CI are recorded in ignored `local_artifacts/r10-v3/release-receipt.json` when verified; absence means not yet confirmed.

This worktree's `.venv` and `local_artifacts` are ignored symlinks to the **already approved public-other-match** runtime/cache in review-pr1. No parent/target directory scan. All new writes are confined to `local_artifacts/r10-v3`; R6–R9 namespaces are read-only except the missing R9 release metadata receipt completed at handoff. Do not copy2.6GB data or redownload SkillCorner20. No chmod/delete/rewrite of historical models or branches.

Frozen code checkpoints: C1 `d72fc4e` (invalid evaluation order); C1FIX and C2 exact SHAs appear in protocol/result row `code_sha` and Git log. Inference source SHA256 and inputs are fixed per protocol. First C1 produced12 invalid score rows; saved proposals/results/logs retained. An attempted stop matched no process (pkill exit1); initial run finished. C1FIX regenerated and rescored under a new name, with sorting regression. No method selection from invalid C1.

Finished: T0 selective source migration; C1FIX12 and C2 12 full-match/domain information probes;48 freshly evaluated full strong-baseline scores in C1FIX, reused in C2; 0 fits/new weights; no final prediction changes. Gate1 failed both mechanisms. Additional short-spell/direction breakdown is posthoc evaluation-only, not a new selection trial. No third geometry threshold search; independent action/tactical observation provenance is the next scientific prerequisite.

Exact read-only recovery command:

```sh
cd /Users/dailuren/Documents/ChatGPT/barca/r10-v3
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/verify_r10_artifacts.py
```

Tests: `.venv/bin/python -m unittest discover -s tests -v`. No data/model access in the synthetic suite. Current release verification logs `/tmp/r10-full-tests.log`, `/tmp/r10-clean-tests.log`, `/tmp/r10-verification.log`; existence and hashes recorded in verification receipt after completion. Logs for experiments `/tmp/r10-c1.log` (invalid), `/tmp/r10-c1fix.log`, `/tmp/r10-c2.log`; pretests logs preserve actual runs. Don't rerun `freeze` over existing protocols or delete output folders to rerun. Runner has exclusive files and hash-checked completed-row recovery. If an interrupted output has NPZ but no receipt, preserve it and create a new explicit experiment ID; don't overwrite it.

All experimental processes have completed before release. No training process is pending; scripts are cache probes. P0 human status remains UNKNOWN (user believes probably not submitted). A must confirm actual email deadline/timezone and manual Form; B must independently run and validate compliant final CSV offline; C confirms public external caches/licenses/exposure. No target file, result, private report or screenshot should be sent back. No automatic merge or formal submission.

Public outputs: FINDINGS, RESULTS_C1FIX/C2 (all per-match metrics), ERROR_CATEGORIES, MODEL_CARDS/REGISTRY, source/contract map, append-only DECISIONS, DELIVERY_MATRIX. Failure and rollback are explicit. Draft publication is separate from scientific success. Exact publication command/receipt is appended at release; never reupload PR4/5 or move their refs.
