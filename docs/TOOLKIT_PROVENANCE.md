# Source, overlap and release scope

Base: `f8f2629ff1e0985f81d7e59d8413678514ee1530` (main).
Selected source: PR #11 / R14 `6b1a78e3d7fccff6cdede7329e007f40269f86f0`.
R11 tools originated in the supplied research toolkit and were preserved in R14.
All selected source is from this MIT repository; the root LICENSE remains intact.
No third-party model code or weights are copied.

| Candidate | Existing equivalent / decision | Local changes |
|---|---|---|
| r11/spell_roi.py | main, PR1 and feat/data already provide scoring/matching; do not duplicate that engine | Thin official adapter; retain validation/edit interface; expose gained/lost GT spells at all five IoUs |
| tests/test_r11_roi_equivalence.py | No equivalent toolkit adapter coverage in main or PR1 | Preserve 1,200 randomized official comparisons and half/X cases |
| tests/test_r11_spell_roi.py | No equivalent single-edit regression suite | Preserve 10 synthetic contracts |
| research_only/oracle_bottleneck.py | No corresponding 12-arm factorial intervention API in reviewed sources | Preserve R14 algorithm; validate finite FPS/full partition; materialize block iterators; copy truth |
| docs/R14_ORACLE_BOTTLENECK.md | Historical R14 evidence, not supplied by main/PR1/feat/data | Retain aggregate external diagnostics; pin original artifact links and distinguish historical runs |
| tests/test_oracle_toolkit.py | Replace mixed R14 tests requiring solution | Independent factorial, provenance, mutation, blocks, random ABC, gain/loss tests |
| r11/__init__.py; research_only/__init__.py | Package plumbing only | Minimal new docstrings |
| TOOLKIT_README.md; this file | New usage/provenance notes | No target configuration or model claims |

Official `evaluation/possession_metrics.py` has the same Git blob
`767494a053628cbdc81a8e9deaf43ef21cc7e58a` in main, PR1 head
`83b262a9420c67eb494b39ed43ece90bd4e66623` and feat/data head
`ff84c79fc8bad6742e22e6bd975a5eef36f4d3e7`.
Its `matched_gt_flags`, `spell_overlaps`, `possession_spells` and `spell_scores`
are reused directly. No second matching algorithm is published.

Read-only PR1 changed-file inventory contains inference/training code and
`tests/test_solution.py`, not an edit ROI or Oracle API. feat/data's generic
source `models/diagnose_possessions.py` (blob
`2c0a074d26c7b4e0a1d410e1ff956fb3819dba91`) already reports short-spell match
loss at IoU .5. It is not copied: its entry point binds target public GT and
prediction paths, while this toolkit accepts external arrays and supports all
thresholds and single-edit gains as well as losses. No referenced GT, predictions
or diagnosis outputs were opened. This is a curated extension, not a claim that
matched-spell analysis itself is novel.

No R12–R15 model, runner, prediction, private material, fixed frame labels,
training configuration or legacy solution module enters this branch. Synthetic
strings are test fixtures only. Historical PRs and local research branches remain
untouched; this branch has only the specified main as parent.

## Release validation (2026-11-17)

A newly created Python 3.12 virtual environment, with no legacy packages, passed
**25 tests** from a separate clean source directory. Installed versions:
NumPy 2.5.5, pandas 3.1.1, python-dateutil 4.0.0, tzdata 2026.6.
This includes 1,200 random official-adapter comparisons, 100 random Oracle ABC
recoveries, and 100 random edit gain/loss conservation checks. Adapter equivalence
checks validate preprocessing/contracts; they are not an independent reimplementation
of the official matching engine. Fixed expected-answer tests supplement them.

The initial sandbox dependency installation failed DNS resolution; a permitted
network installation succeeded. No model or data dependency was added. Exact
reproduction dependency command:

```sh
python -m pip install numpy==2.5.5 pandas==3.1.1 python-dateutil==4.0.0 tzdata==2026.6
python -m unittest discover -s tests -v
```

Official scorer SHA256 remains
`341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d`.
Main was confirmed read-only at the specified base SHA. Review scope is source,
dependencies, selected diff and branch ancestry, not a claim to certify any
unread target/private data. No such data is included or required.
