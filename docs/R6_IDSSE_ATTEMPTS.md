# R6 IDSSE acquisition: actual execution, 2026-11-14

Public other-match data only. No competition assets or public five-minute GT were accessed. Matches J03WMX and J03WN1 remain sealed and were not downloaded.

## Sources and actual fault domains

1. **A — original Figshare:** article `https://api.figshare.com/v2/articles/28196177` and position entry `https://ndownloader.figshare.com/files/51643526`. Initial sandbox DNS resolution failed (curl exit 6). The authorized unrestricted requests actually reached HTTP **403**, including the range probe (118-byte HTML, retained as failed evidence, never treated as XML). The live article title/license/file-table could therefore **not** be independently retrieved here. CC BY 4.0 attribution follows the paper and specified original dataset DOI; it is not a claim of successful Figshare metadata verification.
2. **B — Hugging Face:** `pysport/idsse-data`, fixed revision `a715a38dfbaf5f58e431727c2b78d174101a703c`. Initial direct curl tree request timed out after 15 seconds. A different client (`huggingface_hub==0.36.0`) then successfully obtained J03WPY MatchInformation and full position XML. One position request mistakenly omitted `_observed` and returned 404; floodlight source established the exact filename, corrected once. This was a filename fault, not a mirror outage. J03WOH and J03WQQ metadata and full positions then succeeded on the first correct requests. **No Xet-specific error occurred; `HF_HUB_DISABLE_XET` was not used.** HF is the genuinely independent successful infrastructure.
3. **C — floodlight 1.2.0:** actual `IDSSEDataset(match_id='J03WPY', dataset_dir_name=<absolute ignored directory>)` was attempted. Initial package import failed on missing matplotlib, then PIL. Dependencies were installed only under `local_artifacts/r6/tools`, leaving `.venv` unchanged. The repaired real call reached `urllib.error.HTTPError: HTTP Error 403: Forbidden`, with `socket.setdefaulttimeout(20)`. Its URLs are Figshare ndownloader, so this is **a different client but the same blocked origin**, not a third independent source. No large duplicate file was downloaded. Use the already successful HF files for the independent floodlight parsing cross-check.

Source: Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, [Scientific Data](https://doi.org/10.1038/s41597-025-04505-y); DFL original [Figshare dataset](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0; [pinned mirror](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c). Labels remain the DFL supplier definition, not an asserted equivalence to competition effective-control semantics.

## Validated local bytes

All files are under `local_artifacts/r6/idsse-hf/` and ignored by Git. Prefixes are `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-` and `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-`, then match ID and `.xml`.

| Match/file | Actual bytes | SHA256 |
|---|---:|---|
| J03WPY information | 11,922 | `566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420` |
| J03WPY positions | 371,688,499 | `845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1` |
| J03WOH information | 11,745 | `f23b3ee13b988b1ff9a0bce438c416dac47e71355bbc7ee1dab21aaf3e33c18c` |
| J03WOH positions | 348,945,715 | `ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f` |
| J03WQQ information | 11,932 | `e976aa0c5dba862cfa898ea57ec226efce9d1bc04b23b8d775908db86aa6fa41` |
| J03WQQ positions | 356,219,423 | `cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f` |

J03WPY positions match the supplied original size and MD5 **`e56ff1336a334f8da51110fbe7cab764`** exactly. J03WOH/J03WQQ sizes and SHA256 match the actual pinned HF `get_paths_info` LFS metadata. Their actual MD5s are `3efff25a86a61768e7d253b4bba137a3` and `9105bad987191d3c3c362c1fce9a4fcd`; these are measured mirror-byte hashes, **not independently fetched Figshare MD5 metadata**. Small MatchInformation files are ordinary pinned Git objects, not LFS pointers.

## Reproducible acquisition and cache integrity

```sh
PYTHONPATH=local_artifacts/r6/tools .venv/bin/python scripts/download_idsse.py --match J03WPY
PYTHONPATH=local_artifacts/r6/tools .venv/bin/python scripts/download_idsse.py --match J03WOH
PYTHONPATH=local_artifacts/r6/tools .venv/bin/python scripts/download_idsse.py --match J03WQQ
.venv/bin/python -m unittest tests.test_idsse_download -v
```

The script permits only these three development matches. A complete verified cache is used without any network call. New downloads try Figshare once per file then HF in a separate staging cache; HF internally uses `.incomplete` resumption. Final installation occurs only after exact size, MD5, SHA256 and streaming full XML validation. HTTP resume uses `.part`: 206 must have the correct offset and total; 200 replaces a partial file instead of appending a whole file to it. Verified final caches are never overwritten; corrupt existing final caches fail closed for investigation. Signed URL queries are removed from logged errors.

Twelve synthetic mock tests passed: aligned 206, ignored Range/200, invalid range preservation, corrupted digest rejection, immutable verified/corrupt cache handling, LFS/HTML/wrong XML rejection, completed `.part` promotion, signed-error redaction, fully offline cache acquisition and Figshare-to-HF staged failover and reachable metadata/license conflict rejection. No test downloads real data.

Actual process evidence remains in ignored caches and temporary diagnostic logs: `/tmp/r6-hf-download.log`, `/tmp/r6-hf-position.log` (404), `/tmp/r6-hf-position-correct.log`, `/tmp/r6-hf-j03woh.log`, `/tmp/r6-hf-j03wqq.log`, `/tmp/r6-floodlight-final.log`. Durable successful cache-validation records are appended to `local_artifacts/r6/download_attempts.jsonl`; the table above preserves the important evidence independently of temporary logs.

**Decision: KEEP** the HF recovery and validated public raw caches. Stop repeating the blocked Figshare route; no need to wait for it before training. **Next:** independent floodlight/streaming-parser comparison, then separate-match Alive/Home training under the frozen R6 protocol.

## Independent parser check actually completed

`floodlight.io.dfl.read_position_data_xml(position_path, information_path)` from the isolated **floodlight 1.2.0** installation was run against the downloaded J03WPY XML. Compared with `local_artifacts/r6/prepared/J03WPY/raw.npz`, both halves had exactly equal BallStatus and BallPossession arrays and equal ball XY coordinates (including missing coordinates). Peak RSS was 555,307,008 bytes, approximately 530 MiB. No full-file duplicate download was needed.

| Half | Frames | BallStatus D / Alive | Possession 1 / 2 | Status SHA256 (int8 bytes) |
|---|---:|---|---|---|
| firstHalf | 69,131 | 22,951 / 46,180 | 44,369 / 24,762 | `c1a7fa2e0dc8bb337ea548741eef49ed07d41a9d16777f6a0e40b9ef96081813` |
| secondHalf | 77,080 | 41,800 / 35,280 | 34,825 / 42,255 | `9d43457fed4a3f31e934ec807ae8b1ad41d80004f6eeaa0fd54595c585ca01da` |

Possession int8 byte SHA256: first `03461e426905b19889b469b0585fc22e2f985e438aeaa2f01807572c8a47b412`, second `09c2adda5a355c6b664f04309dbd2e0218fffaafa74ec6fab7cc5d3eaff80c7d`. Equal arrays imply equal transition positions; this establishes parser agreement, not independent confirmation of supplier label semantics. Full comparison result is preserved locally at `local_artifacts/r6/floodlight_crosscheck.json`.

Reproduction after preparing the raw arrays (no network calls):

```python
# PYTHONPATH=local_artifacts/r6/tools MPLCONFIGDIR=local_artifacts/r6/matplotlib .venv/bin/python
from pathlib import Path
import numpy as np
from floodlight.io.dfl import read_position_data_xml
root = Path('local_artifacts/r6/idsse-hf')
xy, possession, status, _, _ = read_position_data_xml(
    root / 'DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml',
    root / 'DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml')
with np.load('local_artifacts/r6/prepared/J03WPY/raw.npz') as raw:
    for index, half in enumerate(('firstHalf', 'secondHalf'), 1):
        mask = raw['period'] == index
        assert np.array_equal(raw['ball_status'][mask], status[half].code)
        assert np.array_equal(raw['ball_possession'][mask], possession[half].code)
        assert np.allclose(raw['ball_xy'][mask], xy[half]['Ball'].xy, equal_nan=True)
```

The isolated optional environment was installed using `pip --target local_artifacts/r6/tools --no-deps`; matplotlib/PIL import dependencies were repaired there only. Frozen `.venv` and V3 model packages were not upgraded. Parser checks need floodlight, lxml, iso8601, h5py and matplotlib (plus its ordinary import dependencies); standard-library acquisition verification and all twelve network mocks need none of these optional packages.

The same full-array comparison subsequently completed **serially for all three development matches**, without network access. Each report records package versions, floodlight parser source hash, the two XML hashes, prepared raw NPZ hash, both half label-array hashes, status/possession counts, and equal transition positions implied by exact labels. Peak RSS across the serial process was **626,319,360 bytes (~597 MiB)**.

| Match | First-half frames | Second-half frames | Status/possession/ball XY match |
|---|---:|---:|---|
| J03WPY | 69,131 | 77,080 | All equal |
| J03WOH | 69,424 | 67,790 | All equal |
| J03WQQ | 69,204 | 73,141 | All equal |

Reports: `local_artifacts/r6/floodlight_crosscheck_J03WPY.json`, `floodlight_crosscheck_J03WOH.json`, `floodlight_crosscheck_J03WQQ.json`. Process log: `/tmp/r6-floodlight-all-crosscheck.log`. Versions: floodlight 1.2.0, lxml 6.1.3, NumPy 2.4.6, iso8601 2.1.0 in the separate tool directory. Floodlight parser source SHA256: `b03cc2432f7ff862daee9a9bf12b767009499927d1a1725e227204e20a581125`. Prepared raw hashes: J03WPY `673abd10e7bb5bbab867595da7d663f967d3df531d7aa12d05f5f6a24d95bf0c`; J03WOH `4fee61d723c90bf470852472a32df3f1e3f96f48092038b8b6f04da2bdc57899`; J03WQQ `fe13bf275ae7d26688bf9097cf34471479cb96374d3a0c30b077d805722f734e`.

**KEEP:** all three parsed development matches passed independent parser agreement. The frozen stress holdouts remain unopened. This removes the raw availability/parser-alignment blocker for actual three-match training; supplier semantics and cross-provider deployment remain separate limitations.
