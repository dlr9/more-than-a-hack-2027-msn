# Public solution development

This retains PR #1's 44 spatial/temporal features, two HistGradientBoosting heads, reference smoothing and Viterbi decoder. V2 repairs parsing, label semantics, provenance, isolation and validation. It is not an approved competition submission.

Use Python 3.12.5 and `pip install -r requirements-solution.txt`. Run `python -m unittest discover -s tests -v`. See [the fixed experiment protocol](../docs/experiment_v2_protocol.md) and [the engineering review](../docs/engineering_review_v2.md).

`python scripts/fetch_external_v2.py` downloads only four explicitly listed public SkillCorner development matches and public Metrica Game 3. `python scripts/experiment_v2.py` prepares and evaluates them. Artifacts stay in ignored `local_artifacts/`. Neither script accesses the competition match or its public GT. Do not point this environment at private target folders.

The default SkillCorner label policy leaves uncovered phase frames unknown; its alive labels have no verified negative class. Therefore ordinary two-head training stops with an explicit missing-class error. The experiment runner deliberately fits a weak phase-complement alive head for diagnostic comparisons only. Those scores are not full H/A/D ground truth evidence.

Feature schema 2 keeps 44 dimensions but renames player counts to `n_home_positions` and `n_away_positions`. Caches declare schema and observation view (`all`, `observed`, or Metrica `available_unknown`); incompatible training caches are rejected. Old caches/models are not silently interchangeable. Use `python scripts/experiment_v2.py --followup` only to reproduce the documented post-hoc HA-only augmentation diagnostic; it does not produce a new test set.

The experimental calibration and control proxy are evaluated in the experiment runner; the legacy `fit` command remains an explicitly uncalibrated diagnostic path.

The target runner is for the human team's isolated offline environment only and requires `--offline-team-ack` before accessing any target path. `final` is disabled pending a complete independent H/A/D gate. Do not run it in cloud Codex. Reference code is preserved; the heuristic no longer treats broad set-piece tags or a hardcoded target frame as proof of dead ball.

The public tree guard examines file names only and stops on suspect report/data paths. It cannot establish that arbitrary prose or old Git history is safe. PR #1's target technical report is excluded from this clean development branch and still requires human review in its original PR/history.
