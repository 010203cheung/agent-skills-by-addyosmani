---
name: health-data-privacy-and-deidentification
description: Guides agents through classifying, minimizing, and de-identifying personal and health-related data before any analysis, modeling, or sharing. Use when a task involves person-level data of any kind — loading a CSV with names or IDs for EDA, building models on patient, student, or customer records, committing datasets to version control, or preparing data to share with collaborators. Use even when the data "isn't medical" — sales, school, and loyalty data become health-adjacent the moment they're linked to health questions.
---

# Health Data Privacy and De-identification

## Overview

Privacy-first data handling for analytics and machine learning. Treat every person-level dataset as identifying until proven otherwise, every identifier as a liability, and every commit of raw data as irreversible. In healthcare data science, a leaked dataset is not a bug to patch — it is harm that cannot be undone. This skill is a gate that runs *before* exploratory analysis, not a cleanup task after.

## When to Use

- Loading any dataset containing names, IDs, contact details, or other person-level records
- Building or evaluating models on patient, student, employee, or customer data
- Linking a non-health dataset (retail, school, F&B) to a health-related question
- Preparing data or notebooks to share with collaborators or domain experts
- Committing a project that involves data to a public (or private) repository
- Publishing results, dashboards, or screenshots derived from person-level data

**NOT for:** aggregate-only public statistics (e.g., national census tables), fully synthetic data you generated yourself, or datasets explicitly released for open use with no person-level records.

## Process: The Privacy Gate

Run these steps in order before any EDA, modeling, or sharing. If a step fails, stop and resolve it — do not proceed "just to explore."

### Step 1: Classify every column

Inventory the dataset and label each column into one of four classes:

| Class | Definition | Examples |
|---|---|---|
| **Direct identifier** | Identifies a person on its own | Name, NRIC/FIN, passport no., phone, email, full address, photo, MRN |
| **Quasi-identifier** | Identifies a person when combined with others | Date of birth, postal code, gender, ethnicity, occupation, admission date |
| **Sensitive attribute** | The private fact you must protect | Diagnosis, medication, test result, disability status, mental health flag |
| **Non-identifying** | Neither identifies nor is sensitive | Aggregated counts, product category, anonymous session metrics |

Write this inventory down (a simple table in the project's data dictionary). If you cannot classify a column, treat it as a quasi-identifier.

### Step 2: Apply data minimization

For each column, ask: *does the analysis question actually need this?*

- Drop direct identifiers that serve no analytical purpose (names never predict outcomes).
- Drop quasi-identifiers not used as features.
- If a column is needed only for joining tables, replace it with a study-specific pseudonym (Step 3) and drop the original.

The safest data is data you never kept.

### Step 3: Pseudonymize what must be joinable

If records must be linkable across tables, replace the direct identifier with a random study ID, and keep the mapping table separate from the analysis data (or destroy it if re-linking is never needed).

```python
import pandas as pd
import uuid

# Build mapping: one random ID per person
mapping = (df[["nric"]].drop_duplicates()
           .assign(study_id=lambda d: [uuid.uuid4().hex[:12] for _ in range(len(d))]))

# Apply, then drop the real identifier
df = df.merge(mapping, on="nric").drop(columns=["nric"])

# Store mapping OUTSIDE the project repo, encrypted, or delete it
mapping.to_csv("/secure/location/id_mapping.csv", index=False)
```

**Do not "anonymize" by hashing national IDs.** NRIC-style identifiers have a small, structured keyspace — an unsalted (and even a salted-but-known-salt) hash can be reversed by brute force in minutes. Random surrogate IDs, not hashes.

### Step 4: Generalize quasi-identifiers

Reduce precision until individuals blur into groups:

- Dates of birth → age, or better, age bands (`0-17, 18-39, 40-64, 65+`)
- Event dates → month/year, or relative days from an index event
- Full postal code → district or region (in Singapore, first 2 digits of the postal code → general area; or planning area)
- Rare categories (an occupation held by one person) → "Other"

```python
df["age_band"] = pd.cut(df["age"], bins=[0, 18, 40, 65, 120],
                        labels=["0-17", "18-39", "40-64", "65+"], right=False)
df = df.drop(columns=["date_of_birth"])
```

### Step 5: Check re-identification risk (k-anonymity quick test)

Every combination of quasi-identifiers should describe at least *k* people (k ≥ 5 is a common working threshold). Groups of size 1 are individuals you have failed to hide.

```python
quasi = ["age_band", "gender", "region"]
group_sizes = df.groupby(quasi).size()
risky = group_sizes[group_sizes < 5]
print(f"{len(risky)} quasi-identifier combinations expose fewer than 5 people")
```

If risky groups exist, generalize further (wider bands, coarser regions) or suppress those rows, then re-run the check.

### Step 6: Keep data out of version control

- Add data directories to `.gitignore` **before** the first commit:

```gitignore
data/
*.csv
*.xlsx
*.parquet
!data/synthetic_sample.csv
!data/data_dictionary.md
```

- Commit only: a **data dictionary** (column names, types, classifications from Step 1) and, if useful, a small **synthetic sample** you generated so others can run the code.
- Clear notebook outputs before committing (`jupyter nbconvert --clear-output`) — printed dataframes leak rows even when the CSV is ignored.
- If raw data was ever committed, deleting the file in a new commit does NOT remove it from history. The data must be treated as exposed; rewriting history (`git filter-repo`) helps only if the repo was never public.

### Step 7: Document the decisions

Add a short `DEIDENTIFICATION.md` (or a section in the README) recording: what was removed, what was generalized and how, the k-anonymity result, and where the mapping table lives (or that it was destroyed). This log is what turns "I deleted some columns" into a defensible process.

## Regulatory Anchors

This skill is jurisdiction-aware but not legal advice. Know which framework governs your data:

- **Singapore — PDPA**: governs personal data generally; the PDPC publishes anonymisation guidance describing techniques (suppression, generalization, pseudonymization) and re-identification risk assessment. NRIC numbers have additional handling restrictions under PDPC advisories.
- **Singapore — Human Biomedical Research Act (HBRA)**: applies when data is used for human biomedical research; institutional review may be required.
- **US — HIPAA Safe Harbor**: de-identification by removing 18 categories of identifiers (names, geographic subdivisions smaller than state, all date elements except year, contact details, IDs, biometrics, photos, etc.) — a useful checklist even outside the US.
- **EU — GDPR**: pseudonymized data is still personal data; only truly anonymized data falls outside scope. Don't claim "anonymized" when you mean "pseudonymized."

When frameworks conflict, apply the strictest one that plausibly applies.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "It's just a portfolio/school project, no one will look" | Public repos are indexed, scraped, and archived within hours. Recruiters *will* look — and what they'd find is mishandled personal data. |
| "I'll de-identify it later, after EDA" | EDA outputs (notebook cells, plots, screenshots) leak rows. Once committed or shared, exposure is permanent. The gate runs first. |
| "I hashed the IDs, so it's anonymous" | Hashing a small structured keyspace (NRIC, phone numbers) is reversible by enumeration. Use random surrogate IDs. |
| "I removed names, so it's anonymous" | 87% of the US population is unique on ZIP + birth date + gender (Sweeney). Quasi-identifiers re-identify; that's why Steps 4–5 exist. |
| "The data isn't medical, it's just sales/school data" | The moment it's linked to a health question, it becomes health-adjacent. Sensitivity comes from use, not just origin. |
| "The dataset is too small to matter" | Small datasets are *easier* to re-identify — every group has low k. |
| "The repo is private" | Private repos get made public, cloned, and shared. Access control is not de-identification. |

## Red Flags

- A CSV with names, NRIC-pattern strings, or emails appears in `git status`
- Notebook outputs showing raw rows of person-level data
- "Anonymized" used to describe data that still has a re-linkable mapping
- Quasi-identifier groups of size 1 in the k-anonymity check
- A join key that is a real-world identifier (phone, NRIC, email)
- No data dictionary or de-identification log in the repo
- Date-of-birth or full postal code retained "just in case"

## Verification

After completing the privacy gate, confirm:

- [ ] Column inventory exists, with every column classified (Step 1 table)
- [ ] No direct identifiers remain in the analysis dataset
- [ ] `grep -rE '[STFGM][0-9]{7}[A-Z]' .` (NRIC pattern) and an email-pattern grep return nothing in tracked files
- [ ] k-anonymity check run; no quasi-identifier group below threshold (evidence: printed output)
- [ ] `git ls-files` shows no raw data files; `.gitignore` covers data directories
- [ ] Notebook outputs cleared in committed notebooks
- [ ] `DEIDENTIFICATION.md` (or README section) documents what was done
- [ ] Mapping table, if kept, is stored outside the repo

If any box cannot be checked with evidence, the dataset is not ready to analyze or share.

