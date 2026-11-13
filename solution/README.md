# Solution pipeline

The implementation keeps the official reference intact and adds three reproducible modes.

Create or update the fixed environment from the repository root:

```powershell
conda env create -f environment-solution.yml
# For the already-created environment:
conda activate hack2027
python -m pip install -r requirements-solution.txt
```

```powershell
conda activate hack2027

# Byte-for-byte reference reproduction
python -m solution.run --mode reference --out evaluation/results/reference_smooth50.csv

# Rule-based candidate; this is not promoted merely because the public five minutes look better
python -m solution.run --mode heuristic --out evaluation/results/heuristic_v1.csv

# Trained candidate after public external data has been prepared and validated
python -m solution.run --mode final --model-dir local_artifacts/models `
  --out evaluation/results/final_candidate.csv
```

External datasets stay outside the repository. Prepare SkillCorner matches and create a fixed
match-level split with:

```powershell
python -m solution.train prepare-skillcorner `
  --root D:\PublicFootballData\skillcorner\data\matches `
  --out local_artifacts\skillcorner

python -m solution.train prepare-metrica `
  --meta D:\PublicFootballData\metrica\Sample_Game_3_metadata.xml `
  --raw D:\PublicFootballData\metrica\Sample_Game_3_tracking.txt `
  --events D:\PublicFootballData\metrica\Sample_Game_3_events.json `
  --out local_artifacts\metrica_game3.npz

python -m solution.train fit `
  --prepared local_artifacts\skillcorner `
  --split local_artifacts\skillcorner\split.json `
  --metrica local_artifacts\metrica_game3.npz `
  --out local_artifacts\models
```

`fit` writes `training_report.json` and marks the model as promoted only when both the
SkillCorner validation rules and the Metrica cross-provider rule pass. `--mode final` refuses an
unpromoted model; `--mode model` remains available for local diagnostics. The held-out
SkillCorner test split is evaluated only after promotion.

The five-minute public truth is a pipeline check only. It is never read by training or model
selection code.

## External data and licensing

- SkillCorner source: <https://github.com/SkillCorner/opendata>. The repository is MIT-licensed;
  credit SkillCorner when publishing work based on the data. Tracking is 10 fps, coordinates are
  in metres around pitch centre, and phases of play exist only while the ball is in play.
- Metrica source: <https://github.com/metrica-sports/sample-data>, Sample Game 3. Its README asks
  public users to acknowledge Metrica Sports; the repository does not currently advertise a
  standard license in GitHub's repository metadata, so re-check its terms before redistribution.
- Competition data remains subject to the competition terms and stays under the ignored `data/`
  tree. Prepared NPZ files, fitted models and candidates remain under ignored local paths.

The implementation deliberately does not consume player IDs, shirt numbers, concrete club names
or a fixed attacking direction as model features. Match IDs are used only to create and audit the
grouped split.

## Freeze record

Before submission, record the following in the team log (do not commit restricted artifacts):

```powershell
git rev-parse HEAD
python --version
python -m pip freeze
Get-FileHash solution\config.json -Algorithm SHA256
Get-FileHash local_artifacts\models\alive.joblib -Algorithm SHA256
Get-FileHash local_artifacts\models\home.joblib -Algorithm SHA256
Get-FileHash <team-name>.csv -Algorithm SHA256
python evaluation\validate_submission.py <team-name>.csv
```

The verified reference SHA256 is
`d444caa1aebb6fb58386033a5967c83302b8dd984ea91c86381bf208332702ff`.

