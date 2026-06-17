---
name: health-data-standards-and-interoperability
description: Guides agents through working correctly with clinical data standards — FHIR and HL7 for exchange, and ICD, SNOMED CT, and LOINC for coding — so health data is parsed, mapped, and joined without silently corrupting clinical meaning. Use when ingesting data from a hospital or EHR system, mapping codes between systems, joining datasets that use different terminologies, or designing how a project will exchange health data. Use even when the data "looks like a normal table" — clinical codes carry meaning a plain join will destroy.
---

# Health Data Standards and Interoperability

## Overview

Correct handling of clinical data standards for analytics and AI. Real health data does not arrive as a clean flat CSV — it arrives as FHIR resources, HL7 messages, and fields populated with coded terminologies (ICD, SNOMED CT, LOINC) whose meaning lives in external code systems, not in the value itself. Treating a diagnosis code as a plain string, or joining two datasets that use different code systems as if the numbers were comparable, silently corrupts clinical meaning while producing output that looks perfectly valid. This skill is the gate that runs at ingestion: it makes sure the data's clinical semantics survive the trip into your analysis.

Core principle: **a clinical code is a pointer into a code system, not a value.** Its meaning depends on which system and version it came from, and that context must travel with it or the data becomes ambiguous.

## When to Use

- Ingesting data from an EHR, hospital system, registry, or health API
- Parsing FHIR resources or HL7 v2 messages into analysis-ready tables
- Mapping or crosswalking codes between systems (e.g., ICD-9 to ICD-10, local codes to SNOMED)
- Joining datasets that may use different terminologies or code versions
- Designing how a POC will receive, store, or exchange health data with a partner
- Interpreting what a coded field actually means before using it as a feature or label

**NOT for:** de-identification of these fields (privacy skill first), or general data-quality issues unrelated to coding (quality skill) — though coded data has standard-specific quality pitfalls covered below.

## Process: The Interoperability Gate

### Step 1: Identify the standard and version before parsing

Establish what you are actually holding, because the parsing strategy depends entirely on it:

- **FHIR** (Fast Healthcare Interoperability Resources): a modern, JSON/XML, resource-based standard (Patient, Observation, Condition, Encounter, etc.) exchanged over REST APIs. The current direction of health interoperability.
- **HL7 v2**: the older but still ubiquitous pipe-and-hat message format (`|` and `^` delimited segments like `PID`, `OBX`). Most hospitals still run on it.
- **C-CDA / other documents**: structured clinical documents.

Record the standard, its version (FHIR R4 vs R5 differ; ICD-9 vs ICD-10 are not interchangeable), and the source system. Version is not a footnote — it changes meaning.

### Step 2: Parse with a real library, never with string slicing

Do not hand-roll parsers for clinical formats. The structures have escaping rules, optional segments, repetition, and nesting that ad-hoc string splitting gets subtly wrong.

```python
# FHIR: use a proper client/parser, then navigate resources explicitly
# (e.g. fhir.resources or fhirclient); pull codes from coding arrays, not raw text
condition_code = condition["code"]["coding"][0]["code"]
condition_system = condition["code"]["coding"][0]["system"]   # which code system!
# HL7 v2: use hl7apy / python-hl7 rather than splitting on '|' and '^'
```

A coded concept in FHIR is a `CodeableConcept` with a `coding` array — each entry pairs a `system` (the URI of the code system) with a `code`. Always read the `system` alongside the `code`; the same number means different things in different systems.

### Step 3: Know which terminology answers which question

The major code systems are not interchangeable; each has a job:

| System | Used for | Example role |
|---|---|---|
| **ICD-10 / ICD-10-CM** | Diagnoses, billing, mortality stats | "What condition?" — administrative/epidemiological |
| **SNOMED CT** | Clinical findings, procedures, fine-grained meaning | "What exactly did the clinician mean?" — rich clinical concepts |
| **LOINC** | Lab tests and observations | "Which measurement is this?" — identifies the test |
| **CPT / procedure codes** | Procedures, billing | "What was done?" |
| **RxNorm / ATC** | Medications | "Which drug?" |

Using the wrong system for a question — e.g., trying to identify a specific lab analyte from an ICD code — produces nonsense. Pick the system that natively answers your question.

### Step 4: Map between systems explicitly, with an authoritative crosswalk

When you must convert (ICD-9→ICD-10, local codes→SNOMED, or harmonizing two sources), use an official mapping (e.g., published GEMs for ICD-9/10, or a recognized terminology service), and treat mapping as lossy:

- Mappings are often **one-to-many or many-to-one** — one ICD-9 code may map to several ICD-10 codes and vice versa. There is rarely a clean 1:1.
- Some concepts have **no exact equivalent** — record these, don't silently drop or force-fit them.
- Keep the original code alongside the mapped code, plus the mapping source and version, so the transformation is auditable and reversible.

Never invent a mapping by eyeballing code similarity; codes that look adjacent can be clinically unrelated.

### Step 5: Respect code hierarchies and granularity

Clinical terminologies are hierarchical, and this matters for analysis:

- ICD codes roll up (e.g., a three-character category contains many more specific codes). Decide deliberately whether to analyze at the specific or rolled-up level, and do it consistently.
- SNOMED CT is a poly-hierarchy with `is-a` relationships; "diabetes mellitus type 2" is a child of "diabetes mellitus." Aggregating without respecting the hierarchy either fragments one condition into many codes or wrongly merges distinct ones.
- Grouping codes into clinically meaningful categories (phenotyping) is a domain decision — validate your groupings with a clinician or an established code set, not by string prefix alone.

### Step 6: Handle the standard-specific quality pitfalls

Coded clinical data has failure modes a generic quality check misses:

- **Units in observations:** a LOINC-identified lab value still carries a unit (and FHIR `Quantity` has a `unit` and `system`); never compare values across records without reconciling units (ties to the data-quality skill's unit checks).
- **Code system version skew:** the same code may have been retired or redefined between versions; a longitudinal dataset can mix versions across years.
- **Local/proprietary codes:** systems often carry local codes not in any standard — flag and map them, don't assume they're standard.
- **Null flavors / placeholders:** standards encode "unknown," "not asked," "not applicable" distinctly; collapsing them all to missing loses information (ties to informative-missingness).

### Step 7: Document the data provenance and mapping decisions

Write `INTEROP.md` recording: the source standard and version, the code systems present, every mapping applied with its source and version and known losses, the granularity/phenotyping decisions and who validated them, and any local codes encountered. This provenance is what lets a reviewer trust that a "diabetes cohort" actually means what it claims — and it's exactly the documentation healthcare data governance expects.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "The codes are just numbers, I'll join on them" | A code is a pointer into a specific code system. The same number means different things across systems; joining on the bare value mixes unrelated concepts. |
| "ICD-9 and ICD-10 are basically the same" | They differ structurally and in granularity; mapping is one-to-many and lossy. Treating them as interchangeable corrupts any trend spanning the transition. |
| "I'll split the HL7 message on the pipe character" | Clinical formats have escaping, repetition, and optional segments. Hand-rolled parsing gets edge cases subtly and dangerously wrong; use a real library. |
| "SNOMED or ICD, doesn't matter which" | Each terminology answers a different question. Using the wrong one for your question yields clinically meaningless features. |
| "I'll map codes that look similar" | Adjacent-looking codes can be clinically unrelated. Mapping needs an authoritative crosswalk, not visual similarity. |
| "I'll roll everything up to the category to be safe" | Granularity is an analysis decision with clinical consequences; over-aggregating can merge distinct conditions. Decide deliberately and validate. |
| "A code is a code regardless of year" | Code systems are versioned; codes get retired and redefined. Longitudinal data mixes versions, and that must be reconciled. |

## Red Flags

- Joining two health datasets on a code value without checking both use the same system and version
- Diagnosis or lab codes stored without recording which code system they came from
- HL7/FHIR parsed by string splitting rather than a dedicated library
- ICD-9 and ICD-10 data combined across the transition with no mapping step
- A code mapping applied with no record of its source, version, or losses
- Lab/observation values compared without reconciling units
- Local/proprietary codes assumed to be standard
- A clinical cohort defined by code prefix with no clinical validation

## Verification

- [ ] Source standard and version identified and recorded (FHIR R-version, HL7 version, ICD-9 vs 10, etc.)
- [ ] Parsing done with a dedicated library; `system` recorded alongside every `code`
- [ ] Correct terminology chosen for the question (ICD/SNOMED/LOINC/RxNorm as appropriate)
- [ ] Any cross-system mapping uses an authoritative crosswalk; originals retained; losses and many-to-one cases logged
- [ ] Granularity/phenotyping decisions made deliberately and validated against a clinician or established code set
- [ ] Units reconciled on observations; code-version skew checked on longitudinal data; local codes flagged
- [ ] Null flavors / placeholders handled distinctly, not collapsed blindly to missing
- [ ] `INTEROP.md` documents standard, code systems, mappings with versions and losses, and validation

If any box lacks evidence, a coded field may not mean what the analysis assumes — and in clinical data, the wrong meaning is indistinguishable from a right answer until it reaches a patient.
