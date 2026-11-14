# R6 provider-neutral feature contract

`solution/provider_neutral.py` adds `provider_neutral_features_v3`. The existing
44 feature names, original preparation functions and frozen schema-2 candidates
are preserved. Schema **3** is required because player speed is consistently
computed from consecutive positions; matching column names alone does not make
old weights compatible. `Candidate.probabilities` already rejects schema changes.

## Raw input and units

```python
from solution.provider_neutral import FeatureContract, provider_neutral_features_v3
contract = FeatureContract(provider="idsse_dfl", observation_view="available_unknown",
                           fps=25., pitch=(105., 68.), coordinate_units="centered_m")
batch = provider_neutral_features_v3({
    "frame": frame, "period": period, "ball_xy": ball_xy,
    "players_xy": players_xy, "player_team": player_team,
    "player_track_id": player_track_id,
}, contract)
# Only after feature extraction, attach separately audited y_alive / y_home.
```

`ball_xy` has shape N×2 and `players_xy` N×P×2. Teams use 0=H, 1=A,
−1=unknown and shape P or N×P. IDs have the same two shape options and only
prevent speed differences across tracklet changes. They never enter the feature
matrix. Without IDs, the adapter explicitly takes responsibility for stable
player slots within each period. Team identity must come from match information,
not possession labels. Rows must be aligned by `(period, frame)` before this API.

Centered metres convert independently as `x / pitch_length + .5` and
`y / pitch_width + .5`; normalized input follows the same physical geometry.
Adapters must establish and document `fixed_pitch_axes` from provider metadata;
the builder does not infer direction from teams or labels. Out-of-pitch points
are preserved, not clipped. Missing coordinate pairs become NaN. Infinite
coordinates, unordered/duplicate within-period frames and unknown units fail.
Reported provider speed (`S` in km/h), ball height and provider detection flags
are not accepted. Speeds use position displacement × FPS in metres/second,
only across consecutive frames in the same half and the same player tracklet.

Temporal means and displacements reset at every period boundary and frame gap.
Masking uses `ball_dropout_mask` on raw positions **before** distances, speed,
nearest-player codes and temporal windows are recomputed. Missing ball is an
observation condition, never a D label. The API never derives labels, interpolates
positions using labels, or treats absent labels as absent positions. The adapter
must preserve raw coordinate missingness independently of label availability.

## Compatibility gate

`assert_compatible(training_contracts, candidate_contract, purpose="train" or
"evaluate")` rejects provider, observation-view, FPS and pitch differences by
default. Deliberate domain-shift experiments require the corresponding explicit
`allow_cross_provider`, `allow_cross_view`, `allow_cross_fps`, or
`allow_cross_pitch` flag. Its returned waiver record belongs in the experiment
manifest. Neither training nor evaluation gets an automatic observation-view
waiver. Schema, feature-definition and orientation mismatches cannot be waived.
Input units can differ because outputs share the same physical unit definition.

Every prepared output carries `feature_contract_json`, schema, provider,
observation view, FPS, pitch, orientation and output units **outside X**.
Call this gate before stacking matches and before artifact evaluation; merely
creating metadata does not enforce downstream caller behavior. In particular,
`observed`, `available_unknown`, and all-position tracking are distinct views.
An explicit waiver permits studying a shift; it is not evidence that target
tracking quality or calibration matches an external provider.

## Leakage boundary and validation

The raw mapping has a strict field whitelist. Labels (`BallStatus`,
`BallPossession`, `y_home`, etc.), events, speed fields and private quality flags
raise an error instead of being silently merged. Arbitrary metadata cannot
become feature columns. No identity, Z, external event type or ground-truth
temporal history is encoded. An adapter can still introduce upstream leakage;
this boundary does not substitute for auditing its alignment and label logic.

The synthetic tests verify two-axis conversion on a non-square pitch, metric
speed, raw-mask equivalence, half/gap resets, identity-invariant features,
schema-2 rejection, label-field rejection and explicit domain-shift waivers:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_provider_neutral.py' -v
```

No target assets, external truth or model weights are used by these tests.
No control-contact feature is added yet: the retained 44-feature baseline allows
the new supervised-data experiment to isolate its main mechanism.
