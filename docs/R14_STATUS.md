# R14 recovery and release status

Current decision: NO_EVIDENCE_TO_PROMOTE. Oracle diagnostic and code retained; all new inference channels disabled. R6–R13 models/cache preserved. No new fits. Worktree /Users/dailuren/Documents/ChatGPT/barca/r14; branch codex/r14-information-gain-20261117; base 49558064e0ebac6be5d1cc27b70523d69be61f92. Final release SHA is the commit containing this file and the ignored publication receipt, avoiding recursive hashes.

Read R14_DECISION.md, R14_P0_AUDIT.md, R14_ORACLE_BOTTLENECK.md, R14_MULTIVIEW_ALIGNMENT.md and R14_FEATURE_INCREMENTAL_VALUE.md. JSONs carry every complete result, frozen protocol, source/model-input SHA and match-gain/loss. R14_MODEL_REGISTRY.json links run commits and prediction hashes. R14_EXPERIMENT_LEDGER.json retains failures and stopping decisions. V1 files are not overwritten by V2.

Executed 72 full external oracle scores with official/ROI agreement; independent saved-artifact check revalidated all 72 score records and 34 unique historical source/input hash combinations. Three actual full-timeline dynamic timing evaluations failed the preregistered incremental gate; no fit followed. Six full natural/long disabled-model evaluations and three switch modes preserve baseline bytes. Eleven synthetic multiview scenarios plus camera null/pan/cut checks are engineering validation only. Full test suite: 257 local tests passed. Clean-source test result is recorded in R14_VERIFICATION.json once completed.

Remote metadata main f8f2629ff1e0985f81d7e59d8413678514ee1530 descends through c073f29098efb8b44bf06725b09c4ad59f0f990f, 3de6b60aeabd2475d95879a651d38b72bc4fba00, f00bc32a6bb7457b1ff28815e37606115daf252e, b42969321c74e07c97972d271fffc3bd01405bb6 to root e47adea397fc6f1a51de48b7c44dc5bf9ba1a927. Research lineage has root 974216d (allowlisted public snapshot). Consequently publish an incremental independent Draft PR against codex/r13-control-recovery-20261116, not main. Do not merge unrelated histories or import target-bearing teammate assets. GitHub publication does not submit the competition Form.

## Exact recovery (read-only default)

```sh
cd /Users/dailuren/Documents/ChatGPT/barca/r14
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2
.venv/bin/python -m unittest discover -s tests
# Choose a NEW receipt path; old receipts are never overwritten.
.venv/bin/python -m scripts.verify_r14 --out /tmp/r14-verification-new.json
```

Python 3.12 / NumPy 2.4.6 / SciPy 1.17.1 / sklearn 1.7.2. Existing .venv and local_artifacts symlinks refer to the protected public-external cache in ../review-pr1. No runtime download. Do not point these research commands at official target inputs or output files.

Fresh reproducibility environment ONLY (retain upstream R6/R8/R12 public inputs, exclude R14 output directory and generated R14 result reports; never delete the originals):

```sh
.venv/bin/python -m scripts.research_r14_oracle run
.venv/bin/python -m scripts.research_r14_multiview run
.venv/bin/python -m scripts.research_r14_dynamic run
```

Do not refreeze protocols on a rerun. Original protocol source versions are preserved at 861cea9 (Oracle), 7847baa (multiview V1), cfc8b10 (dynamic V1), fd1e7ca (multiview/dynamic repairs). V1 dynamic WPY was completed and retained; WOH/WQQ resumed under V2. Writers reject existing result paths. To reproduce each historical protocol exactly, use its matching commit in a separate worktree; fresh V2 dynamic all-three rerun is expected to preserve numerical metrics, but do not relabel it as the original mixed-version run.

Disk exhaustion and unverified image downloads remain recorded, no historical cache pruned. Recovery scripts do not autonomously retry downloads, install heavyweight dependencies or consume sealed matches. Rollback is simply leaving new channels off and retaining HC_M3/M3. A/B/C responsibilities and the single next evidence/data gate are in R14_DECISION.md.
