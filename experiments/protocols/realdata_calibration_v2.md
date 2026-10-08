# Protocol: real-data calibration study v2 (pre-registered)

Written and committed **before** any run of this study. The git commit that adds this file is the
time stamp.

## Why
The first real-data study (`run_realdata.py`, 200 splits per condition) observed V6 clean-alarm
rates of 2.0–2.5% on COMPAS, but with 200 splits the exact upper bounds (5.04–5.74%) could not
show the < 5% gate. Following the project rule *"decide the number of repetitions and all gates
before seeing outcomes; do not append seeds to the old study"*, this is a **new, fixed-size study
with fresh seeds**. It is not an extension of the old one.

## Fixed design
- Code: the auditor code at the commit that adds this file. No change to `cfa/` during the study.
- Methods: V6 (primary) and V5 (secondary, B = 1,999), both on every split (paired).
- Datasets and conditions (no injected corruption in any of them):

| dataset | condition | audit sampling |
|---|---|---|
| COMPAS | clean | uniform |
| COMPAS | shift_age_1.0 | probability ∝ exp(1.0·z_age) |
| COMPAS | shift_age_1.5 | probability ∝ exp(1.5·z_age) |
| German Credit | clean | uniform |
| German Credit | shift_age_1.0 | probability ∝ exp(1.0·z_age) |

- Splits: 25% clean test (unused here), 30% of the remainder trusted, 70% of what is left audit
  (same as the first study).
- Repetitions: **1,000 splits per condition** (5,000 in total), fixed in advance.
- Seeds: `5_000_000 + 100_000·dataset_index + 10_000·condition_index + rep`. This range is
  disjoint from every earlier study (3.0M real-data, 7.0M calibration, 9.0M power, 11.0M effect size).

## Endpoints and decision rule
- Primary: V6 clean-alarm rate (share of splits with any flag) per condition, with the exact
  two-sided 95% Clopper–Pearson interval.
- Gate G1 (real data): the upper bound is below 5% in **every** one of the five conditions.
  If any condition fails, the paper reports "G1 not demonstrated on real data" and keeps the
  failing condition in the results.
- Secondary: V5 alarm rates (same gate); paired exact McNemar test V5 vs V6 per condition;
  which blocks were flagged in false-alarm splits (descriptive).
- Every condition is reported whatever the outcome. No condition may be dropped or re-run.

## Command
```bash
python experiments/run_realdata_calibration.py --reps 1000 --jobs 10
```
