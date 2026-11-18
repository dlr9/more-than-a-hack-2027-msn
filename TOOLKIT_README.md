# Possession evaluation toolkit (research only)

This small toolkit extends the **unchanged official scorer**, not the inference
pipeline. It adds single-edit matched-spell gains/losses and privileged external
label diagnostics. It does not provide a new model or claim a leaderboard gain.

## Install and reproduce

From the repository root, Python 3.10+:

```sh
python3 -m venv .venv-toolkit
.venv-toolkit/bin/pip install numpy pandas
.venv-toolkit/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Only the official scorer's NumPy/pandas dependencies are needed. No `solution`,
model weights, downloads, videos, label CSVs or training dependencies are needed.

```python
from r11.spell_roi import score, audit_edit
from research_only.oracle_bottleneck import ExternalTruth, interventions
import numpy as np

gt = np.array(list('HHADDDAHHAA'))  # synthetic example only
pred = np.array(list('HHHAAADDDAA'))
blocks = [(0, len(gt))]
print(score(gt, pred, blocks=blocks).as_dict())
print(audit_edit(gt, pred, 2, 3, 'A', blocks=blocks))
arms = interventions(pred, np.linspace(.1, .9, len(gt)),
                     ExternalTruth(gt, 'synthetic'), blocks, fps=25)
for name, labels in arms.items():
    print(name, score(gt, labels, blocks=blocks).spell_f1_mean)
```

## Contract and interpretation

Frame indices are zero-based; every interval is half-open `[start,end)`. Inputs
must already share one frame grid, team identities and period boundaries. The
caller supplies actual FPS (positive, finite), not assumed 25 FPS. Home inputs
are frozen conditional `P(H | Alive)`, not joint H probability. Oracle blocks
must partition the complete timeline; adjacent half blocks remain separate.
Scoring accepts H/A/D/X and intersects blocks with non-X truth. Predicted X is
filled **within** each scored run as in the official temporal evaluation.
D is not a possession spell. Five IoUs are .3/.4/.5/.6/.7, greedily matched by the
official implementation. Edit results identify newly matched and lost **GT
spells** at every threshold; they do not claim stable prediction-pair identities.

Oracle A grants the true D mask; B grants true direct H/A transition times AND
count, with model direction; C grants majority ownership on proposed intervals.
D_short grants true short-spell existence, endpoints AND owner. The 12 arms are
privileged, order-dependent, nonadditive diagnostics, not deployable models,
causal attribution or attainable performance bounds. They may lose matches.

Use **only legal other-match complete truth or synthetic fixtures**. Never feed
competition target assets, private truth, the official five-minute GT, or target
derivatives into these tools, nor use them for target tuning/model selection.
`ExternalTruth` accepts `synthetic` and `IDSSE_DFL_source_complete`; this declaration
is a misuse check, not proof of provenance or a security sandbox. The caller must
verify permission and match identity before loading any labels. There is no
automatic file loader. Freeze real proposals before label-based edit evaluation.

Historical development-only evidence and its limitations are recorded in
[the R14 report](docs/R14_ORACLE_BOTTLENECK.md). This release does not rerun or
independently validate those matches; it verifies the extracted code contracts.
[Source and duplication review](docs/TOOLKIT_PROVENANCE.md) records what was kept.

## Rollback

No existing tracked main file, default model or inference entry point changes.
Switch back to `main` to remove this additive toolkit from the working tree.
This candidate is local only; publishing requires user approval, and merging
requires separate approval.
