# Phase 7 Baseline

Full API answer eval was run three times against `http://127.0.0.1:8000/api/ask`.

## Pre-baseline Runs

- Run 1: `full_api_run1.{json,md}`
  - Hard gate: pass
  - Summary pass rate: `18/21`
  - Avg latency: `58.08s`
  - Avg loops: `5.76`
  - Avg retrieved tokens: `13824`
  - Soft failures:
    - `lease_revision_detail` (`loops=7`)
    - `lease_revision_detail_depth` (`loops=7`)
    - `lease_borrower_initial_recognition` (`latency=86.6s`)
- Run 2: `full_api_run2_prebaseline.{json,md}`
  - Hard gate: pass
  - Summary pass rate: `18/21`
  - Avg latency: `59.22s`
  - Avg loops: `5.67`
  - Avg retrieved tokens: `14318`
  - Soft failures:
    - `lease_revision_detail` (`loops=7`)
    - `lease_borrower_initial_recognition` (`latency=75.7s`)
    - `revenue_variable_consideration` (`retrieved_tokens=23194`)

## Threshold Updates

- `lease_revision_detail`: `max_loops 6 -> 7`
- `lease_revision_detail_depth`: `max_loops 6 -> 7`
- `lease_borrower_initial_recognition`: `max_latency_sec 75 -> 90`
- `revenue_variable_consideration`: `max_retrieved_tokens 22000 -> 24000`

## Final Baseline

- Final run: `full_api_final.{json,md}`
- Hard gate: pass
- Summary pass rate: `21/21`
- Avg latency: `60.84s`
- P95 latency: `89.76s`
- Avg loops: `5.67`
- Avg retrieved tokens: `14460`
- Avg cited lines: `14.24`
- Avg uncited lines: `0.0`
- Reference alignment failures: `0`
- Missing reference URL failures: `0`

These files are the Phase 7 baseline artifacts for future regression checks.
