---
name: health-data-quality-validation
description: Guides agents through profiling, validating, and documenting the quality of person-level and health-adjacent datasets before any analysis or modeling. Use when loading any new dataset, merging tables from different sources, inheriting data collected by someone else, or whenever results look suspiciously good or bad. Use even when the data "looks clean" — health-adjacent data hides impossible values, unit mix-ups, sentinel codes, and duplicate people that silently corrupt every downstream conclusion.
---

# Health Data Quality Validation

## Overview

Systematic data validation for analytics and machine learning on health-adjacent data. A model trained on corrupted data produces confident, well-formatted, wrong answers — and in healthcare, wrong answers carry clinical weight. The most dangerous datasets are not the obviously broken ones; they are the ones that load without errors. This skill is a gate that runs after the privacy gate and before any EDA conclusion, feature engineering, or modeling.

Core principle: **never edit raw data by hand.** All cleaning is code, applied to an immutable raw copy, producing a derived clean copy — so every correction is visible, reviewable, and reversible.

## When to Use

- Loading any dataset for the first time
- Merging or joining tables from different sources or time periods
- Inheriting data collected by someone else (domain partners, exports, surveys)
- Before reporting any statistic or training any model
- When results look too good (suspect leakage or duplicates) or too bad (suspect corruption)
- After any manual data entry process touched the data

**NOT for:** privacy and de-identification decisions (use `health-data-privacy-and-deidentification` first), or fixing upstream collection processes themselves (raise those with the data owner — but document them here).

## Process: The Quality Gate

### Step 1: Profile before touching anything

Generate a profile of the raw data and READ it before cleaning:

```python
import pandas as pd

df = pd.read_csv("data/raw/visits.csv")

print(df.shape)
print(df.dtypes)
print(df.describe(include="all").T)        # min/max/uniques per column
print(df.isna().mean().sort_values(ascending=False))  # missingness per column
print(df.head(10))
```

Look specifically at: min and max of every numeric column (impossible values live at the extremes), number of unique values in categorical columns (typo variants inflate them), and columns whose dtype is `object` but should be numeric (a stray text value forced the whole column to strings).

### Step 2: Validate plausibility ranges

Every health-adjacent numeric column has a physically or logically possible range. Encode the ranges as explicit rules, not mental notes:

| Variable | Hard limits (reject) | Soft limits (flag for review) |
|---|---|---|
| Age | < 0 or > 120 | > 105 |
| Height (cm) | < 30 or > 250 | < 120 or > 210 for adults |
| Weight (kg) | <= 0 or > 500 | < 30 or > 200 for adults |
| Heart rate (bpm) | < 0 or > 300 | < 40 or > 180 at rest |
| Systolic BP (mmHg) | < 0 or > 350 | < 80 or > 220 |
| Body temperature (°C) | < 25 or > 45 | < 35 or > 41 |

```python
hard_violations = df[(df["age"] < 0) | (df["age"] > 120)]
assert hard_violations.empty, f"{len(hard_violations)} impossible ages — inspect before proceeding"
```

Hard violations are data errors: quarantine the rows (export them for the data owner), never silently delete. Soft violations may be real outliers — a 2-meter-tall patient exists — so flag and review, don't auto-remove. Deleting true extremes biases health data badly, because the extremes are often the patients who matter most.

### Step 3: Hunt unit mix-ups and sentinel values

Two silent killers in merged health data:

**Unit mix-ups.** Weight recorded in kg in one source and lbs in another; height in cm vs meters; glucose in mmol/L vs mg/dL (a ~18x difference). Detection heuristic: plot the distribution — unit mix-ups appear as two separate clusters (a bimodal histogram of "weights" clustering near 60 and near 150 usually means kg and lbs mixed).

```python
df["weight"].hist(bins=50)   # bimodal? suspect mixed units
```

**Sentinel values.** Legacy systems encode "unknown" as 999, -1, 0, or 9999 instead of leaving blanks. A blood pressure of 999 averaged into your mean corrupts every statistic downstream.

```python
suspects = df["sbp"].value_counts().head(20)   # sentinel codes show up as spikes
df["sbp"] = df["sbp"].replace({999: pd.NA, -1: pd.NA, 0: pd.NA})
```

Also normalize categorical encodings: `"Y"/"N"/"1"/"0"/"TRUE"/"yes "` (note the trailing space) in the same column must be collapsed to one scheme before any group-by.

### Step 4: Check date logic

Dates fail logically, not just syntactically. Test the relationships:

```python
df["admit"] = pd.to_datetime(df["admit"], errors="coerce")
df["discharge"] = pd.to_datetime(df["discharge"], errors="coerce")

assert (df["discharge"] >= df["admit"]).all(), "discharge before admission found"
assert (df["birth_date"] <= pd.Timestamp.today()).all(), "future birthdates found"
```

Also check: age column vs computed age from birthdate (mismatch means at least one is wrong), event dates outside the study period, and the classic day/month swap (a suspicious absence of days > 12 means dates were parsed in the wrong order).

### Step 5: Resolve duplicates — carefully

Three kinds, three different treatments:

1. **Exact duplicate rows** (same values everywhere): usually an export or merge artifact. Safe to drop — but count and log them first.
2. **Same person, multiple records, different values**: may be data entry duplicates (typo in one copy) — or may be *legitimate repeat visits*. In health data, repeat encounters are the norm, not an error. Dropping them indiscriminately deletes real clinical history; keeping them naively makes one frequent patient dominate statistics. Decide explicitly: person-level analysis → aggregate to one row per person; visit-level analysis → keep all, but never let the same person appear in both training and test sets (this is leakage).
3. **Same person under different identifiers** (typo'd ID, name spelling variants across sources): hardest to detect; check with fuzzy matching on quasi-identifiers, and flag rather than auto-merge.

```python
exact = df.duplicated().sum()
per_person = df["study_id"].value_counts()
print(f"{exact} exact duplicate rows; {(per_person > 1).sum()} people with multiple records")
```

### Step 6: Treat missingness as information, not just absence

Before filling or dropping anything, ask WHY values are missing. The standard vocabulary:

- **MCAR** (missing completely at random): a sensor glitch — missingness unrelated to anything. Safe-ish to drop or impute.
- **MAR** (missing at random given observed data): older patients skip an online survey question — missingness explained by other columns. Imputation can work if done carefully.
- **MNAR** (missing not at random): the value itself caused the gap — the sickest patients were too unwell to attend the follow-up. Dropping or naive imputation here *biases the result in the worst direction*.

Health data is full of *informative missingness*: a lab test that was never ordered often means the clinician judged it unnecessary — the gap itself carries signal. Practical default: add a missingness indicator column before imputing, and never silently `dropna()` the whole table.

```python
df["lab_x_missing"] = df["lab_x"].isna().astype(int)   # keep the signal
```

Report missingness per column in the quality report; a column > 40–50% missing usually should be questioned, not imputed.

### Step 7: Produce a data quality report and keep raw immutable

- Raw files live in `data/raw/` and are never edited; cleaning code reads raw and writes to `data/clean/`. Re-running the pipeline from raw must reproduce the clean data exactly.
- Write `DATA-QUALITY.md` recording: row/column counts before and after, hard violations found and quarantined, sentinel codes replaced, unit corrections applied, duplicate decision and counts, missingness summary and the chosen treatment, and known issues left uncorrected (with reasons).
- Encode the checks as assertions in the pipeline so they re-run automatically when data updates — a quality check that runs once is a snapshot; one that runs every time is a guardrail.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "The data loaded fine, so it's clean" | Loading tests syntax, not truth. A heart rate of 999 loads perfectly. |
| "I'll just dropna() and move on" | If missingness is MNAR (the sick missed follow-up), dropping rows silently deletes the most important patients and biases every conclusion. |
| "Outliers should be removed" | In health data, extremes are often the real, clinically critical cases. Remove impossible values; review extreme ones. |
| "I fixed a few cells by hand in Excel" | Manual edits are invisible, unrepeatable, and unreviewable. All cleaning is code on an immutable raw copy. |
| "Duplicates are errors, drop them all" | Repeat visits are legitimate clinical history. Dropping them deletes data; keeping them naively causes leakage and skew. Decide per analysis level. |
| "My model gets 99% — the data must be fine" | Suspiciously good performance is a data-quality symptom: duplicates straddling the train/test split, or a leaked answer column. |
| "Cleaning is boring, the model is the real work" | Practitioners estimate the majority of project time goes to data preparation — because it determines whether the model means anything. |

## Red Flags

- A numeric column stored as `object`/string dtype
- Bimodal distribution in a single-unit measurement (mixed units)
- Spikes at 0, -1, 999, or 9999 in a clinical measurement (sentinels)
- Discharge dates before admission dates; no day-of-month values above 12
- `dropna()` or `drop_duplicates()` called without a logged count of what was removed
- Raw data files modified in place; no `data/raw/` vs `data/clean/` separation
- Mean or std reported on a column that still contains sentinel codes
- Same `study_id` appearing in both training and test sets

## Verification

- [ ] Profile output (shape, dtypes, describe, missingness) reviewed and saved
- [ ] Plausibility rules encoded as assertions; hard violations quarantined with a count, not silently dropped
- [ ] Distributions plotted for key measurements; no unexplained bimodality
- [ ] Sentinel values identified and converted to proper missing values
- [ ] Date logic assertions pass (ordering, no future dates, age consistency)
- [ ] Duplicate counts logged; person-vs-visit level decision documented
- [ ] Missingness summarized per column; treatment chosen and justified; indicator columns added where informative
- [ ] `data/raw/` untouched; pipeline reproduces `data/clean/` from raw
- [ ] `DATA-QUALITY.md` exists and records every correction and every known issue left as-is

If any box lacks evidence, conclusions drawn from this data are opinions, not findings.
