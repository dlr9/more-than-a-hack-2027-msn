# External development evidence — 2026-11-14

These are other-match development results. **No candidate is promoted.** SkillCorner HAD labels are weak phase proxies; Metrica labels cover only event interiors. Full spell F1 is not reported. See [protocol](experiment_v2_protocol.md) and [all aggregate metrics](external_validation_v2.json).

## HA cross-fit macro F1 (four matches)

| Domain | Variant | Calibrated | Mean | Match bootstrap 95% interval |
|---|---|---|---:|---|
| skillcorner | all | False | 0.9229 | [0.9017, 0.9475] |
| skillcorner | observed | False | 0.9183 | [0.9029, 0.9335] |
| skillcorner | dropout | False | 0.9176 | [0.9023, 0.9324] |
| skillcorner | all | True | 0.9224 | [0.9015, 0.9472] |
| skillcorner | observed | True | 0.9182 | [0.9027, 0.9335] |
| skillcorner | dropout | True | 0.9175 | [0.9024, 0.9329] |
| dropout_stress | all | False | 0.8786 | [0.8620, 0.8958] |
| dropout_stress | observed | False | 0.8898 | [0.8708, 0.9028] |
| dropout_stress | dropout | False | 0.8880 | [0.8789, 0.8967] |
| dropout_stress | all | True | 0.8781 | [0.8596, 0.8954] |
| dropout_stress | observed | True | 0.8902 | [0.8708, 0.9032] |
| dropout_stress | dropout | True | 0.8854 | [0.8790, 0.8954] |

Intervals resample four development matches; they are not strong generalization evidence. Paired deltas and intervals are in the JSON.

## Metrica partial HAD diagnostic

Each row averages predictions from four separately trained fold models on the **same** Metrica game. This is descriptive model variability, not four-match external validation.

| Variant | Calibrated | HA macro F1 | Fixed HAD macro F1 | Control-proxy HAD macro F1 | Reference HAD macro F1 |
|---|---|---:|---:|---:|---:|
| all | False | 0.8989 | 0.8680 | 0.8679 | 0.8721 |
| all | True | 0.8989 | 0.8725 | 0.8725 | 0.8721 |
| observed | False | 0.8859 | 0.8697 | 0.8696 | 0.8721 |
| observed | True | 0.8842 | 0.8729 | 0.8729 | 0.8721 |
| dropout | False | 0.8868 | 0.5104 | 0.5104 | 0.8721 |
| dropout | True | 0.8852 | 0.5105 | 0.5105 | 0.8721 |

## Label coverage audit

| Match | In-period frames | Phase-gap frames | HA supervised frames | HA missing-ball frames | Gap activity overlap, >1s from boundaries |
|---|---:|---:|---:|---:|---:|
| 1874553 | 60282 | 27526 | 19507 | 195 | 0 |
| 1886347 | 59042 | 27028 | 18453 | 265 | 47 |
| 1899585 | 60512 | 30831 | 15793 | 141 | 0 |
| 1927964 | 59052 | 21891 | 23985 | 113 | 0 |

Dynamic-event overlap is same-provider evidence and does not independently establish ball state. Gap duration, visibility and overlap details are in the JSON. No trustworthy D labels are inferred from absence.

## Calibration and missing-ball strata

JSON rows contain both heads’ Brier score, log loss, ten-bin ECE/reliability and HA missing-ball counts/metrics. Synthetic dropout is a stress test with preserved source labels, not evidence that naturally missing-ball labels are representative.

## Reproduction and disposition

Fixed protocol, source hashes and failures are documented in [engineering review](engineering_review_v2.md). No hyperparameter/model was selected using these results. The final output gate remains closed.
