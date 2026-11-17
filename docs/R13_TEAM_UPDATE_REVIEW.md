# Team update review — 2026-11-17

Remote metadata confirms `feat/data` advanced from a54219dbff8948507b112234127816c133f7c08b to ff84c79fc8bad6742e22e6bd975a5eef36f4d3e7. Research PR9 remains Draft at e22a2a5fbc847b2d14fce55e8593eb91d5a7db0c.

The tree inventory contains new V5 source plus target prediction/report paths. Those output/report contents were NOT opened. Only allowlisted `models/fusion_v5.py`, `models/test_fusion_v5.py`, and `run_pipeline.py` were requested as source. Unexpectedly, source/test code itself embeds target-specific label intervals. Review stopped at that boundary. None of these intervals, label values or target statistics are reproduced here, downloaded into this worktree, executed, or used in experiments. Team must isolate that material before a clean generic-source review can continue.

Safe static findings:

- `apply_event_refinement` writes fixed match-specific output slices. It is not a general visual or whistle inference function. Its regression test checks those fixed outputs, which does not demonstrate external-match validity.
- The inference path does not call the anomaly detector and does not run a visual/audio model. The advertised multimodal stages should not be interpreted as implemented learned sensor evidence.
- The generic out-of-pitch latch has state across the entire supplied timeline. Applying match blocks only afterward does not reset its hidden state at half/gap boundaries.
- A single observed out-of-bounds coordinate can activate the latch. Missing observations cannot establish a restart, so a detection outlier followed by occlusion can become an arbitrarily long false D interval. Whole-ball line crossing, coordinate uncertainty, slow restarts and different FPS/pitch contracts require explicit treatment.
- The runner couples prediction with automatic public-target evaluation. Research and authorized local submission need separate entry points; target GT must not become an implicit model-selection loop.

No claim about V5 target score, team intent, or formal competition-rule violation is made. Its target-specific implementation cannot be adopted under this project's cloud isolation and model-selection constraints. No direct merge or teammate branch modification occurred.

## Direction decision

R13 actually trained a dense event model and tested a structural correction, not another audit-only round. Neither added reliable complete-spell gains. In training-match frames where the event expert dominates, its direction agreement with full source HAD is only about77%; on two evaluation matches it is about75%, below the frozen baseline on those same frames. This does not support assuming that a bigger version of the same geometric model will solve control acquisition.

Stop the existing geometry/penalty family. The promising different route is independently observed action evidence, with uncertainty: a real visual contact/control or referee-audio detector, clean automatic event outputs, and a segment model that treats contact as insufficient for ownership. However no paired legal full-match video/HAD inference result exists in this environment, and target material cannot be used to fill that gap. OSL remains a concrete externally sourced candidate, not a demonstrated solution; inherited R12 source/compute/overlap gates remain applicable.

A second prerequisite is an external semantic audit of receipt/contact versus durable control. Human event timing and DFL source possession disagree at some receipt times. Neither should be silently treated as equivalent to official controlled acquisition. A new evaluation corpus must retain complete H/A/D and ambiguous non-events; provider event anchors alone cannot certify a system that autonomously detects all changes.

Today is the stated submission date. Authorized team members must separately confirm exact timezone/deadline, valid offline CSV, public reproducible code, technical documentation and Form receipt. Current submission status remains UNKNOWN because no Form receipt or authorized offline validation is available; the user also does not know the status. No cloud target run or submission was performed.
