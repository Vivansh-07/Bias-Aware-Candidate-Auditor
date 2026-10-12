# Paper revision notes (10 Oct 2026)

Original: the team's Overleaf draft (a single `main.tex`). Revised: `main.tex` and `main.pdf` in this folder.
Both compile with IEEEtran. The original runs to 7 pages; the revision is exactly 6, with no overfull boxes and no
undefined references.

## Checked against the raw results in tanistheta/Bias_detection

| Claim in the paper | Status |
|---|---|
| Table III (MAE and bias, benchmarks and ACS round 1) | matches exactly |
| Table IV (per-target reduction, CIs, means 7.3 / 14.5 / 23.9) | matches |
| Table V (ACS guard, both rounds) and pooled harm rates | match |
| Fig. 2 (7-target means 10.6 ... 18.5, oracle 27.0, best DF 6.9, MNAR line) | match |
| Table VI (H1-H7, R1-R3) | matches |
| ACS: -4.1% vs 8.1%, oracle 31-58%, Spearman 0.85, harm 38/60% vs 15/5% | match |
| T2 vs T1: 3.7-5.3 worse (benchmarks), 2.7-6.8 better (ACS) | match |
| Disclosure bias: 3.0-5.1 (benchmarks), 7.4-13.7 (ACS) | match |
| AUC cost at most 1.3 points; neural check oracle did not improve | match |
| Pilot: uncertainty/minority/diverse 9.6-11.4% vs 13.8% random | **9.7**-11.4% (fixed) |
| "ACS targets had 20 to 23 candidate groups" | **19** to 23, counting A (fixed) |
| Table I and "Spearman 0.84", "1.3e-14" | **see below** |

## The one real problem: Table I used a different model

The paper says the audited model is logistic regression with lambda = 1e-3 (Fig. 1, Section V). But the stored
benchmark audit (results/e1_audit.csv) only reproduces with lambda = 1e-4, the pilot value. That is why Table I and
Table IV gave different gaps for the same model (Adult 0.14 vs 0.16, German 0.18 vs 0.17, COMPAS-sex 0.15 vs
0.14). The ACS audit already used 1e-3 and matches the main experiment exactly.

Re-running the repository's own `scripts/exp_audit.py`, which already defaults to 1e-3, keeps every conclusion:
510 audits, A ranked first in 0, identity exact to 1.2e-14. Some Table I values change, and now agree with Table IV:

| Target | EO gap of A | Rank of A | abs(rho_sA) | max abs(rho_sG) |
|---|---|---|---|---|
| Law race | 0.39 | 6 of 8 | 0.44 -> **0.45** | 0.73 -> **0.74** |
| German age | 0.18 -> **0.17** | 16.5 of 38 -> **37** | 0.14 | 0.53 -> **0.54** |
| Credit age | 0.16 -> **0.17** | 12 of 22 | 0.14 | 0.73 -> **0.74** |
| Adult sex | 0.14 -> **0.16** | 14 of 18 | 0.23 -> **0.24** | 0.65 -> **0.67** |
| COMPAS sex | 0.15 -> **0.14** | 5 of 5 | 0.13 | 0.55 |
| OULAD disability | 0.11 | 15.5 -> **16** of 24 | 0.09 | 0.51 |
| COMPAS race, Dutch sex | unchanged | | | |

Section IV now reads Spearman 0.85 (was 0.84) and 1.2e-14 (was 1.3e-14). The Fig. 1 ranges now read 0.44-0.74 and
0.03-0.45. The matching repository change (re-run audit plus the missing script that builds e1_summary.csv) is in
the pull request from Vivansh-07/Bias_detection.

## Other edits
- **Abstract.** Shortened and rewritten for grammar. It now uses the pre-registered 8-target numbers (14.5% vs 7.3%, as in H1
  and Table IV). The old 15.5% / 6.9% are 7-target means, and Section VI-B still gives them, labelled as such.
- **Introduction.** Tightened, and the contributions list was shortened. The claims are unchanged.
- **Methods III-D.** Now states that the pilot compared five strategies and the confirmatory runs kept two, as
  PREREGISTRATION.md says. The old text implied that all five were confirmatory arms.
- **Significance.** "50% / around 85% / also 85% / 37.5% targets" became "4 of 8, 6 of 7, 7 of 8, 3 of 8", after
  Holm correction.
- **Discussion.** The sentence fragment and the missing spaces after full stops are fixed, and so is the wording
  "how well does the results transfer".
- **Layout.** Table II no longer runs 11.5 pt into the margin (scriptsize). Table I uses tighter column spacing, and
  Table VI has wider, ragged-right columns, which needed `\usepackage{array}`.
- **Authors.** "Dr." was removed before Chirag Joshi, because IEEE author lines do not use titles. Put it back if the team
  prefers.
- **Spelling.** "labeled/labeling" became "labelled/labelling" throughout, matching the rest of the text.

## Still open (needs the team)
1. **Pre-registration record.** Sections III-G and V and the Data Availability statement say that the gap guard
   tau = 0.08 was fixed before any ACS run, and that R1-R3 were pre-registered before ACS round 2. They also say the
   pre-registration is in the repository. PREREGISTRATION.md has neither section. Tanishk should add the dated
   text he wrote before those runs. Without it a reviewer cannot check the claim, and the repository was created
   on 9 Oct, so its history cannot timestamp it.
2. **Missing scripts.** The README points to `make_readme_figures.py` and `get_references.py`, and neither is in the
   repository. The script that produced `e1_summary.csv` is also missing; the pull request adds a replacement.
3. **Hosted app (done 10 Oct).** Data and Code Availability now links the interactive demonstration at
   https://hidden-group-bias.streamlit.app. The paper is still exactly 6 pages.
