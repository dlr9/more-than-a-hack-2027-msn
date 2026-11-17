# R14 Oracle bottleneck: diagnostic, never model performance

Executed 72 complete external evaluations (3 matches × natural/long × 12 interventions), each checked against the unchanged official five-IoU scorer and ROI scorer. ABC exactly reconstructs the supplied external truth as a sanity check. These previously consumed DFL matches are development diagnostics, not blind validation. Long is artificial label-independent occlusion; natural ball-missing coverage is zero.

| Match/domain | HC_M3 | True D | True HA times/count | True owner on proposed segments | Restore true <1s | Restore true <5s |
|---|---:|---:|---:|---:|---:|---:|
| J03WPY/natural | 0.573158 | 0.574278 | 0.736842 | 0.654227 | 0.660419 | 0.902818 |
| J03WPY/long | 0.480720 | 0.508416 | 0.626415 | 0.566526 | 0.574790 | 0.836443 |
| J03WOH/natural | 0.584968 | 0.591795 | 0.757532 | 0.692826 | 0.661836 | 0.847112 |
| J03WOH/long | 0.506765 | 0.516981 | 0.672877 | 0.607616 | 0.585332 | 0.770806 |
| J03WQQ/natural | 0.596818 | 0.614392 | 0.756889 | 0.683834 | 0.647281 | 0.821843 |
| J03WQQ/long | 0.489049 | 0.492897 | 0.612980 | 0.587328 | 0.539837 | 0.738901 |

Natural-match means: baseline .584981; true D .593488 (+.008507); true HA timing/count .750421 (+.165440); segment owner .676962 (+.091981); complete true <5s spells .857258 (+.272277). Long-domain means are .492178, .506098, .637424, .587157 and .782050 respectively. Short-spell intervention reveals existence, endpoints AND team, so it is a larger information grant, not a fair model comparison or a mathematically attainable ceiling.

At IoU .5, baseline subsecond recall is only 6/57, 6/46 and 6/37; 1–5s recall is 74/196, 81/188 and 79/175. Timing-only-in-name B actually supplies event count as well as time and removes prior HA cuts; it retains frozen model direction. It gains/loses 60/13, 46/15 and 45/14 true matches at IoU .5. Even oracle timing loses some formerly matched spells through changed segmentation: a boundary intervention is not automatically harmless.

The A→B→C factorial is order-dependent. Natural mean A+B .769151 versus additive expectation .758928; B+C .979374 versus additive .842402. Direction and timing interact strongly. Do not sum these as independent error budgets. C uses frame-majority ownership, not spell-optimal assignment, and no arm is guaranteed to improve every threshold. Full per-IoU TP/FP/FN, match gains/losses, duration recall, false-D cuts, D misses, typed boundary errors, same-team restarts and observation strata are in R14_ORACLE_BOTTLENECK.json. No hand-picked successful slice is substituted for complete scoring.

Primary inference: invest in independent evidence of short sustained control and acquisition timing, not another bulk Dead-mask or smoothing experiment. This does not prove that visual evidence is available or useful. No oracle enters an inference import or published default. External truth is accepted only in research_only; its source tag is a misuse check, not a security boundary.

Protocol V1 failed a synthetic ABC equality test before scoring because merged B labels discarded latent proposal cuts. V2 retains the candidate intervals for C. V1, V2, source commits and prediction hashes are preserved; no completed score was overwritten. Original runs record exact Git SHA in every row.
