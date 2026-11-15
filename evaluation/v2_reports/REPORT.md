# Fusion V2 experiment report

All public labels have historical tuning exposure. Blocked results and bootstrap are not independent full-match validation.

| Model | spellF1 | macroF1 |
|---|---:|---:|
| baseline_raw | 0.0378 | 0.5972 |
| baseline_smooth125 | 0.4242 | 0.5812 |
| baseline_smooth25 | 0.2868 | 0.6200 |
| baseline_smooth50 | 0.3256 | 0.6207 |
| model_fusion_v1 | 0.4000 | 0.6834 |
| model_state_machine | 0.3745 | 0.6674 |
| fusion_v2 | 0.5935 | 0.7261 |

Default promoted: True

## Ablations

| Removed | spellF1 | macroF1 |
|---|---:|---:|
| no_repair | 0.5935 | 0.7261 |
| no_tags | 0.5000 | 0.7269 |
| no_dead_evidence | 0.4276 | 0.5114 |
| no_temporal | 0.1031 | 0.6760 |

## Validation details

Selection requires fitting-window macroF1 at least equal to V1 before maximizing spellF1. This constraint was added after the unconstrained winner reduced macroF1; the original 120 candidates were reused without additional search.

Public spellF1 bootstrap 5-95%: 0.4129 to 0.7742.

| Held-out block | spellF1 | macroF1 |
|---|---:|---:|
| 1 | 0.5333 | 0.5742 |
| 2 | 0.4667 | 0.5563 |
| 3 | 0.7000 | 0.6892 |

Each held-out row uses parameters selected on the other two blocks. Selected gap=0 disables trajectory repair, so its ablation is identical.

## Full-match checks

| Half | Dead fraction | Switches | Median segment seconds |
|---|---:|---:|---:|
| 1 | 0.413 | 174 | 11.36 |
| 2 | 0.474 | 167 | 10.32 |

Full diagnostics, per-class errors, per-IoU scores, transitions, bootstrap intervals, all trials and fold-selected configurations are in experiments.json.