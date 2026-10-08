# Bias-Aware Candidate Feature Auditor (V6 working model)

**DSE3170 Project Based Learning-3 · Manipal University Jaipur**
Project: *Bias-Aware AI Pipelines with Fairness Detection*
Team: Soumya Shradha · Tanishk Gangwar · Yagya Salwan · Vivansh Garg · Guide: Dr. Chirag Joshi

A runnable implementation and interactive web app for the project's **V6 cross-fitted,
source-adjusted candidate feature auditor**. Before any model is trained, it checks whether an
observed feature (or block of related features) is linked to unusual label behaviour in an audit
dataset compared with a small trusted reference. It **flags** a block only when the evidence
survives multiple-testing correction, and otherwise **abstains**. The auditor is never told which
columns are sensitive.

**Try it online:** <https://bias-aware-candidate-auditor.streamlit.app>. No installation is
needed. If nobody has used the app for a while, Streamlit puts it to sleep; click the wake-up
button and give it about a minute.

> This repository rebuilds the V5 and V6 auditors from the method described in the project's
> report and IEEE manuscript (V1–V6). It is an independent reimplementation. Its numbers come
> from its own simulation studies (see [`docs/RESULTS.md`](docs/RESULTS.md)) and are reported
> separately from the archived V1–V6 study results.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # web app at http://localhost:8501
pytest                        # 37 tests (library + headless app)
```

Command-line auditing:

```bash
python -m cfa --synthetic "Label corruption (30%)"
python -m cfa --dataset COMPAS --corruption 0.3
python -m cfa --csv my_data.csv --label outcome --source source --json report.json
```

`--csv` expects one file with the features, a binary label and a `source` column
(0 = trusted reference, 1 = audit sample).

## What the app shows

| Page | What you can do |
|---|---|
| 🏠 Overview | The problem, the three questions (ranking, detection, mitigation) and the V6 pipeline diagram |
| 🧪 Synthetic Lab | The paper's generator. Run a clean control, label corruption, a clean population shift or random labels; compare V6 with the V5 permutation auditor; reveal the ground truth |
| 🌍 Real-Data Audit | COMPAS, German Credit, Heart Disease or your own CSV. Inject corruption into a chosen group and/or a clean population shift, audit, then train downstream models to see whether acting on the flag improves fairness |
| 📊 Calibration | Live Monte-Carlo alarm rates with exact Clopper–Pearson bounds, plus the saved large studies |
| 📖 Method & Limits | The V1→V6 evolution, the V6 statistic and what a flag or an abstention does *not* mean |

### Five-minute demo script

1. **Synthetic Lab → "Matched clean control" → Run.** Both auditors abstain. The chart still
   ranks blocks (some cross the naive p = 0.05 line), but none survive BY correction.
   *Ranking ≠ detection.*
2. **"Label corruption (30%)" → Run.** V6 flags `{a, p}`: the group and its proxy, grouped into
   one block by the 0.60 association rule. The group chart shows audit labels in `a=1` about
   18 pp lower than trusted after adjustment. Open **Reveal ground truth**.
3. **Calibration → condition `shift_1.50_nonlinear` → Run.** This is a clean population shift with
   a nonlinear label rule. V5 raises false alarms on roughly one dataset in five; V6 stays near
   the nominal rate. *This is why V6 models source membership.*
4. **Real-Data Audit → COMPAS, 30% corruption on `race = African-American` → Run.** V6 flags
   `race`. Then click **Train & evaluate**: audit-guided reweighting removes most of the model's
   over-prediction for that group, close to the clean-label oracle.
5. **Real-Data Audit → Heart Disease.** With only 303 rows the support rules fail and the auditor
   abstains. It refuses to guess instead of producing an unsupported flag.

## How V6 works

```
trusted (S=0) ──► 25% design subset ──► association blocks (≥ 0.60) + group boundaries
      │                                             │ (geometry only, no analysis labels)
      ▼                                             ▼
remaining trusted + audit (S=1) ──► 2 folds per source ──► cross-fitted logistic models
                                                           e(X)=P(S=1|X),  m(X)=P(Y=1|X)
r = (S − e(X)) · (Y − m(X))  ──►  pairwise group contrasts of mean r (normal approx.)
   ──►  Bonferroni within block  ──►  Benjamini–Yekutieli across blocks
   ──►  FLAG supported blocks with q < 0.05, otherwise
        "No statistically supported candidate detected."
```

* The **source model** absorbs clean population differences between trusted and audit data.
  The **label model** absorbs legitimate feature–label relationships. A group-linked
  source–label association that remains is a signal worth reviewing.
* Support rules: ≥ 20 records per source per group, ≥ 10 per source per fold, ≥ 100 analysis
  records per source. Unsupported contrasts keep p = 1.
* Effect size shown per group: θ_g = Σ r / Σ (S − e)², the adjusted audit-minus-trusted
  difference in label probability.

The **V5 comparator** (`cfa/v5.py`) is the coherent-score auditor with conditional source
permutations within (block pattern × risk stratum) cells: hypergeometric draws, B = 1,999, and the
block maximum recomputed in every draw.

**Mitigation** (`cfa/mitigation.py`) runs only after a flag. For the flagged feature's groups, a
label model trained on the trusted reference estimates the clean positive rate in the audit data.
Audit records are reweighted toward that rate (clipped to [0.5, 2], ESS ≥ 70%). The comparison
covers baseline, drop-block, trusted-only and clean-label oracle models, scored on a clean
held-out test set (AUC, EO gap, DP gap, group prediction bias).

## Repository layout

```
app.py                     Streamlit web app (the working model)
cfa/
  stats.py                 BY / BH / Holm, Clopper–Pearson gates
  geometry.py              design-subset groups, association blocks, encodings
  v6.py                    V6 cross-fitted residual auditor
  v5.py                    V5 coherent-score permutation auditor (comparator)
  simulate.py              synthetic generator + calibration conditions + demo presets
  realdata.py              COMPAS / German Credit / Heart loaders, semi-synthetic splits
  mitigation.py            audit-guided reweighting + downstream fairness evaluation
  __main__.py              command-line interface
experiments/
  run_study.py             calibration + power Monte Carlo (paired V5/V6)
  run_realdata.py          real-data transfer + mitigation study
  make_report.py           figures + docs/RESULTS.md
results/                   saved study outputs (per-run records and summaries)
docs/RESULTS.md            generated results with figures
tests/                     37 pytest checks (CI numbers from the report, auditors, headless app)
data/                      compas_clean.csv, german_credit.csv, heart_disease.csv
```

Reproduce every number:

```bash
python experiments/run_study.py --study calibration --reps 500
python experiments/run_study.py --study power --reps 500
python experiments/run_realdata.py --reps 200
python experiments/make_report.py
```

## Key results (this reimplementation)

Full tables and figures: [`docs/RESULTS.md`](docs/RESULTS.md). All numbers come from this
repository's own runs. Methods were applied to the same datasets (paired), and every rate carries
an exact Clopper–Pearson 95% interval.

**1. Calibration: 10,500 synthetic datasets, 21 conditions × 500.**
V6 passed **20/20 clean conditions** (alarm rates 0.6–3.0%, largest upper bound 4.90% in the
833-trusted-record condition) and the random-label control (495/500 abstentions). V5 passed 18/20.
It failed the **clean shift + nonlinear label rule** condition with **115/500 alarms (23.0%)**,
where V6 had 12/500 (2.4%), and the small-trusted-sample condition (upper bound 5.14%).

![calibration](docs/figures/fig_calibration.png)

**2. Power: 7,500 datasets.** This is the first V6 power study; the archived V6 stopped before
power testing. Recall of the corrupted block at n_audit = 5,000:

| corruption | 10% | 20% | 30% | 40% |
|---|---|---|---|---|
| V6 recall | 3–5% | 43–47% | 88–92% | ≥ 99.6% |

The ≥ 80% target is met at 30–40% corruption but **not at 20%**. At 20% it needs
n_audit = 10,000 (83.8%). Under a population shift with 30% corruption, V6 recall is 81.6% versus
93.4% for V5: V6 gives up some sensitivity to stay calibrated.

**3. Real-data transfer: 2,200 record-disjoint splits of COMPAS and German Credit.**
On COMPAS with 30% injected corruption (non-recidivist African-American defendants relabelled
as recidivists), V6 recovered `race` in **59.5%** of splits. V5 recovered it in **0.5%**, because
V5's support rule requires every race group (including the rare merged Asian/Native-American
group) to have ≥ 10 records per source in each risk stratum, so `race` is always untestable for V5.
German Credit (1,000 rows) is mostly too small: V6 recall is 4% at 30% corruption and 35.5% at
50%.

**3b. Pre-registered real-data calibration: 5,000 fresh splits.** The first real-data run
(200 splits) could not settle the false-alarm gate: COMPAS had 5/200 alarms, upper bound 5.74%.
So a new fixed-size study was pre-registered: protocol
[`experiments/protocols/realdata_calibration_v2.md`](experiments/protocols/realdata_calibration_v2.md)
was committed before the first run. **V6 passed the < 5% gate in all five conditions**:

| condition | V6 alarms | upper bound | V5 alarms |
|---|---|---|---|
| COMPAS clean | 2.1% | 3.19% | 0.9% |
| COMPAS age shift 1.0 | 1.8% | 2.83% | 0.8% |
| COMPAS age shift 1.5 | 1.7% | 2.71% | 1.2% |
| German Credit clean | 0.6% | 1.30% | 0.1% |
| German Credit age shift 1.0 | 0.4% | 1.02% | 0.0% |

V5 alarms less often on clean COMPAS (paired p = 0.012), partly because it cannot test `race`.
V6's false alarms come mostly from the skewed juvenile-count features (24 of 58 flagged blocks).

**4. Audit-guided mitigation (COMPAS, 30% corruption, splits where V6 flagged).** Reweighting
cut the logistic model's over-prediction for the corrupted group from **+14.3 pp to +1.6 pp**
(clean-label oracle: +0.2 pp). The equalized-odds gap fell from 0.65 to 0.39, and AUC *rose* by
0.016. The report's acceptance criteria (EO gap down, AUC loss ≤ 0.02, ESS ≥ 70%) were met in
97.5% (LR) and 98.4% (RF) of flagged splits.

![mitigation](docs/figures/fig_mitigation_compas_corrupt_30.png)

## Limits (read before presenting)

* A flag is an investigation prompt, not proof of corruption, discrimination or causation. An
  abstention does not certify the data as unbiased.
* The auditor needs a trusted reference sample and observed informative features (the group
  itself or a proxy). It cannot find a group that is completely unobserved.
* V6 uses a normal approximation and logistic nuisance models. Calibration has to be checked by
  simulation, as the saved studies do. On real data it has been checked on two datasets only
  (COMPAS, German Credit) and is not guaranteed for arbitrary data.
* Real-data results are semi-synthetic: real features, injected corruption. Detecting real,
  naturally occurring label bias has not been validated.

## Relation to the team's other repositories

* [`tanistheta/bias_awareness`](https://github.com/tanistheta/bias_awareness): the PBL-2 ABS / Rawness pipeline (German Credit)
* [`Soumya-205/Bias_Awareness_Research`](https://github.com/Soumya-205/Bias_Awareness_Research): its cross-domain validation and fixes; the three datasets here come from that repo
* [`YAGYASALWAN/Bias-Aware-Banking-Preprocessor-`](https://github.com/YAGYASALWAN/Bias-Aware-Banking-Preprocessor-): banking preprocessor roadmap

Data sources: ProPublica COMPAS analysis (Broward County), UCI Statlog German Credit, UCI
Cleveland Heart Disease.
