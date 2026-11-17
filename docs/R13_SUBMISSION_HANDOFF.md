# Submission readiness — 2026-11-17

Source: [official SUBMISSION.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md), Git blob ef703837ba6855dba010704256d7375752a559f5, checked November 17.

Official requirements: UTF-8 CSV with frame,label header; 199319 data rows for consecutive frames 0 through 199318; labels H/A/D/X and no blanks. Run the official evaluation/validate_submission.py in the authorized isolated environment. Deliver the CSV, a public reproducible repository with pinned dependencies and commands, and a technical PDF/Markdown documenting methods, assets, external training data/labels, limitations and exact reproduction. Repository must not distribute match assets.

The submission Form URL is emailed to registered participants. Registered team name and a Google account are needed. One response per team, editable until the deadline; last pre-deadline response is used. Public instructions specify November 17, 2026 but no precise time or timezone. There is no public live leaderboard. Target-specific manual output labels must not replace algorithmic predictions.

Status:
- FORM_URL/REGISTERED_TEAM/EXACT_DEADLINE_TIMEZONE: UNKNOWN in this session.
- FORM_SUBMITTED/RECEIPT: UNKNOWN, not claimed complete.
- TARGET_ASSETS/RUNNER_VALIDATED/CSV_VALIDATED: UNKNOWN; cloud does not inspect them.
- PUBLIC_RESEARCH_CODE: R13 Draft delivery prepared, not a selected finalist model.
- FINAL_CODE_CSV_DOCUMENT_CONSISTENCY: not verified; must refer to the exact local generating algorithm, not this failed candidate.

A scoped Gmail search for competition registration/submission messages failed with connector Unknown tool. No mail was sent, attachments opened, or Form submitted. No target CSV, predictions or private experiment reports were inspected. A source-only teammate review unexpectedly encountered embedded target-specific output intervals and stopped; those contents are not reproduced.

Recovery for authorized local team: recover the registration message and Form; use a clean generic model revision without hardcoded target outputs; run and validate entirely offline; document the exact algorithm and external training labels; verify public code reproduces the CSV; submit and retain the Form receipt. Research must not delay this. Do not upload target data to Codex to resolve these status items.
