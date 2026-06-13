---
name: imbalanced-data-and-clinical-metrics
description: Guides agents through choosing honest evaluation metrics and handling class imbalance for health-adjacent classification, where the positive case (disease, readmission, fraud) is rare. Use whenever building or evaluating any classifier, comparing models, or reporting performance. Use even when accuracy "looks great" — on imbalanced health data a high accuracy usually means the model learned to ignore the patients who matter. This skill defines the baseline, the right metric, and the threshold BEFORE training.
---

# Imbalanced Data and Clinical Metrics

## Overview

Honest evaluation for classification on health-adjacent data. In healthcare the event you care about is usually rare — a disease affects a small fraction of those screened, readmissions are the minority, fraud is a sliver of claims. On such data, accuracy is actively misleading: a model that predicts "everyone is healthy" scores high and helps no one. This skill is a gate that runs BEFORE training: it fixes the baseline, the evaluation metric, and the decision threshold up front, so results can't be quietly rationalized afterward.

Core principle: **choose the metric for the cost of being wrong, not for the number that looks best.** In health, a missed case (false negative) and a false alarm (false positive) carry very different real-world costs, and the metric must reflect which one hurts more.

## When to Use

- Building, tuning, or comparing any classification model
- Reporting model performance to anyone (a README, a stakeholder, an interviewer)
- The positive class is rare (screening, diagnosis, readmission, adverse events, fraud)
- Deciding the operating threshold for a probabilistic model
- Whenever a reported accuracy looks suspiciously high

**NOT for:** regression problems (continuous outcomes — different metrics), or data-quality issues that fake good performance via leakage (use `health-data-quality-validation` first — a "great" model is often a quality bug, not a metric one).

## Process: The Evaluation Gate

Run steps 1–4 BEFORE training. Steps 5–7 run during and after.

### Step 1: Measure the imbalance and set the baseline

Before any model, compute the class balance and the majority-class baseline. This baseline is the number every model must beat to mean anything.

```python
prevalence = y.mean()                      # fraction positive
majority_baseline_acc = max(prevalence, 1 - prevalence)
print(f"Positive prevalence: {prevalence:.1%}")
print(f"Always-predict-majority accuracy: {majority_baseline_acc:.1%}")
```

If the positive class is 6% prevalent, a do-nothing model scores 94% accuracy. Any model's accuracy must be read against that 94%, never against 100%. State the baseline in the report before stating the model's score.

### Step 2: Choose the metric from the cost of errors

Pick the primary metric deliberately, based on which error is worse:

| Situation | Worse error | Prioritize |
|---|---|---|
| Screening for a serious treatable disease | Missing a case (FN) | **Recall / sensitivity** |
| Flagging for an expensive/invasive follow-up | False alarm (FP) | **Precision** |
| Both matter, classes imbalanced | — | **F1, or PR-AUC** |
| Ranking/triage, threshold not yet fixed | — | **PR-AUC** (precision-recall) |

Accuracy is almost never the right primary metric for imbalanced health data. Name the chosen metric and the reason in the report up front, so the choice can't be reverse-engineered to flatter the result later.

### Step 3: Prefer PR-AUC over ROC-AUC when positives are rare

ROC-AUC is the reflexive default, but on heavy imbalance it is optimistic: because true negatives dominate, the false-positive rate barely moves, so the ROC curve looks good even when the model floods the positive predictions with false alarms. The precision-recall curve (PR-AUC) focuses on the positive class and tells the more honest story when prevalence is low.

```python
from sklearn.metrics import average_precision_score, roc_auc_score
print("PR-AUC :", average_precision_score(y_true, y_scores))   # report this
print("ROC-AUC:", roc_auc_score(y_true, y_scores))             # context only
```

Report PR-AUC as primary; include ROC-AUC for comparability, not as the headline.

### Step 4: Read the confusion matrix, not a single number

Always produce the confusion matrix and the four cells by name — they are the ground truth a single metric compresses away.

```python
from sklearn.metrics import confusion_matrix, classification_report
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(classification_report(y_true, y_pred, digits=3))
```

In a clinical frame: **false negatives are missed patients** (often the gravest harm), **false positives are unnecessary follow-ups** (cost, anxiety, capacity). Discuss both explicitly; a metric that hides the FN count is hiding the harm.

### Step 5: Handle imbalance honestly (and only on training data)

Techniques, from least to most invasive — try the simplest first:

1. **Class weights** (`class_weight="balanced"`): tell the model to penalize minority errors more. No data fabrication; usually the first thing to try.
2. **Threshold tuning:** the 0.5 cutoff is an arbitrary default. Move the threshold to hit the recall/precision your use case needs.
3. **Resampling** (oversample minority / undersample majority / SMOTE): use sparingly, and follow two iron rules below.

Two rules that, if broken, invalidate results:

- **Resample only the training fold, never the test/validation set.** The test set must keep the real-world prevalence, or your metrics describe a world that doesn't exist.
- **Resample inside cross-validation, not before it** — otherwise synthetic or duplicated minority rows leak across folds and inflate scores. Use a pipeline so the resampler is fit per fold.

```python
# Correct: resampling lives inside the CV pipeline (imbalanced-learn)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
pipe = Pipeline([("smote", SMOTE(random_state=42)), ("clf", model)])
# cross_val_score(pipe, X, y, scoring="average_precision") -> SMOTE re-fit each fold
```

### Step 6: Set the threshold for the clinical cost, then report at that threshold

A probabilistic model outputs scores; the threshold turns scores into decisions and is a *policy choice*, not a statistical one. Choose it for the cost trade-off (e.g., "we accept 3 false alarms to avoid missing 1 case"), document the rationale, and report precision/recall/confusion matrix at that operating point — not at the silent 0.5 default.

### Step 7: Reproducibility and the limitations note

- Set and record a random seed (`random_state=42`) everywhere stochastic — splits, resampling, model init — so results are reproducible.
- Split before any preprocessing or resampling (no leakage), group-aware if the same person has multiple rows (see the data-quality skill).
- Write the evaluation into the report: prevalence, baseline, chosen metric and why, PR-AUC and ROC-AUC, the confusion matrix at the chosen threshold, the imbalance technique used, and a limitations paragraph (what the metric does not capture, calibration not assessed, external validity unknown).

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "95% accuracy, the model is great" | If prevalence is 5%, predicting 'all healthy' also scores 95% and helps nobody. Accuracy without the baseline is meaningless. |
| "ROC-AUC is 0.92, so it's strong" | On rare positives, ROC-AUC flatters because true negatives swamp the false-positive rate. Report PR-AUC. |
| "I used SMOTE and accuracy jumped" | If you resampled before splitting or the test set was rebalanced, the jump is leakage, not skill. Resample train folds only. |
| "0.5 is the natural threshold" | 0.5 is an arbitrary default, not a clinical decision. The threshold encodes the cost of FN vs FP and must be chosen on purpose. |
| "Precision and recall are both fine to report either way" | They answer different questions; the right one depends on whether a miss or a false alarm is worse. Choosing reveals you understand the problem. |
| "I'll just optimize accuracy in the grid search" | Optimizing the wrong metric tunes the model toward the wrong behavior. Score CV on PR-AUC / recall, matching the real objective. |
| "The model beats the baseline by a hair, ship it" | A marginal gain over the majority baseline may not justify the false-alarm burden. Compare against the cost, not just the baseline number. |

## Red Flags

- Accuracy reported as the headline metric on imbalanced data
- No majority-class baseline stated alongside model performance
- ROC-AUC reported while PR-AUC is omitted on a rare-positive problem
- Resampling (SMOTE/oversampling) applied before the train/test split, or to the test set
- A single metric reported with no confusion matrix
- Threshold left at 0.5 with no justification
- No random seed; results change run to run
- Performance discussed without ever naming the false-negative count

## Verification

- [ ] Class prevalence and majority-class baseline computed and stated before model results
- [ ] Primary metric chosen and justified by the cost of FN vs FP, in writing
- [ ] PR-AUC reported as primary on rare-positive problems; ROC-AUC for context only
- [ ] Confusion matrix produced; TP/FP/FN/TN discussed by clinical meaning
- [ ] Any resampling confined to training folds, inside CV; test set keeps real prevalence
- [ ] Operating threshold chosen for the cost trade-off and documented; metrics reported at that threshold
- [ ] Random seed set and recorded everywhere stochastic; split done before preprocessing
- [ ] Limitations paragraph present (what the metrics do not capture; calibration; external validity)

If any box lacks evidence, the reported performance is a number, not a finding — and on imbalanced health data, the wrong number is worse than none.
