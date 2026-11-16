# Public evidence audit, 2026-11-16

No target assets or derived results were accessed. No visual model was claimed to have run. Model installation cannot substitute for a paired legal full-match evaluation dataset.

## OpenSportsLab

[Model card](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e), revision `54e24a663a5a0ed057ef12cf16d9460d270ddc6e`. Actual API metadata and 2,733-byte config downloaded; checkpoint NOT downloaded. `model.pt` is 152,081,691 bytes; advertised LFS SHA256 `2352f17f60de586d7531b07f88ae4d618b46c1f8a4904837b0493ea15767307c`. AGPL-3.0 card license. SoccerNet ball-action spotting, RGB 224x398, 100-frame clips; configuration lists input 25 FPS and extraction 2 FPS. This is not possession ownership or full H/A/D output.

[Runtime documentation](https://opensportslab.github.io/opensportslib/getting-started/installation/) describes CPU support with compatible DALI configurations converted to OpenCV. Do not misreport this as GPU-only. Current safe runtime has no Torch and only approximately 0.3 GB free disk; checkpoint plus dependencies cannot be safely staged while preserving frozen artifacts. No approved full public video paired with complete H/A/D is currently staged. Training/target match overlap is UNKNOWN; broadcast-to-tactical transfer is unverified. Status: RESOURCE_AND_PAIRED_DATA_BLOCKED. Web raw-file requests failed; bounded fallback via small HF API/config requests succeeded. No repeated bulk downloads.

## WACV 2026 CropDrop

[Primary paper](https://openaccess.thecvf.com/content/WACV2026/html/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.html), [full text](https://arxiv.org/html/2512.19528v1). Uses player image crops and player trajectories; proprietary tactical matches. Out-of-play frames are excluded, so its reported task does not validate independent Dead detection. Possessor-anchored ball reconstructions cannot become independent observation evidence for the same labels. A downloadable checkpoint, complete match-overlap inventory, and applicable weights license were not established from the inspected primary sources. Status: UNVERIFIED_WEIGHTS_AND_LICENSE. No model inference performed.

## PathCRF correction

Previously preserved original checkpoint and upstream source remain unchanged. Source audit finds roster padding/player_mask support in dataset preparation. Thus strict complete-22-player coverage is an adapter feasibility measurement, NOT proof that the architecture can never handle padding. However SetLSTM/SetTransformer do not establish a verified per-frame missing-identity mask contract; NaN episodes may be skipped. Arbitrary dynamic-slot absence is not equivalent to fixed roster padding. No Torch reinstall or forward-inference success is claimed. Existing 5 FPS limitations and absent verified goalkeeper/direction contract remain.

## ELASTIC: new supervision, not inference oracle

[Repository and license](https://github.com/hyunsungkim-ds/elastic/tree/cikm2026): code MPL-2.0; manually corrected public DFL event benchmark CC BY 4.0. Attribution: Kim et al., ELASTIC, CIKM 2026; Bassek et al., An integrated dataset of spatiotemporal and event data in elite soccer, 2025.

Downloaded ONLY `benchmark/gt/J03WPY.parquet`, 37,462 bytes, git blob `1125bb08c765f0570814adb07fdf01b4622c2b0a`, SHA256 `8ba1df509b68f1e2081b520f87f43ad39218436538023de2abeb31287bfc6dad`. Verified source-frame alignment and audited 1,202 event records using isolated duckdb 1.4.1; no changes to the historical training runtime. Results: `R12_EXTERNAL_EVENT_AUDIT.json`. Synchronization requires an event sequence; these annotations cannot be provided to an autonomous inference model at test time. They offer an additional human timing target for future dense event supervision, not a new blind match or demonstrated gain.
