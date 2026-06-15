---
name: model-explainability-for-healthcare
description: Guides agents through making model predictions interpretable, auditable, and fair enough to act on in a clinical or health-adjacent setting. Use when a model will inform a human decision, when a stakeholder asks "why did it predict that?", when checking a model for bias across patient subgroups, or before presenting any model as a deliverable. Use even when the model is accurate — in healthcare an unexplained correct prediction a clinician cannot trust or justify is not deployable.
---

# Model Explainability for Healthcare

## Overview

Interpretability, auditing, and fairness for models that inform health decisions. A model in healthcare rarely acts alone; it advises a human who is accountable for the outcome. That human needs to know *why* a prediction was made, whether the reasoning is clinically sensible, and whether the model treats patient subgroups fairly. This skill is a gate that runs after the evaluation gate and before any model is presented or deployed: it turns a performant black box into something a clinician can interrogate, a regulator can audit, and a patient can be treated by safely.

Core principle: **a prediction a human cannot understand is a prediction a human cannot safely act on.** Accuracy earns a model a hearing; explainability earns it the right to influence a decision.

## When to Use

- A model's output will inform a human decision (triage, risk flag, resource allocation)
- A stakeholder, clinician, or interviewer asks "why did it predict that?"
- Auditing a model for bias or unequal performance across subgroups (age, sex, ethnicity)
- Choosing between a simpler interpretable model and a complex black-box one
- Before presenting any model as a portfolio deliverable or to a domain partner

**NOT for:** the upstream gates — privacy, data quality, and honest metric selection each have their own skill and must come first. Explainability of a model trained on leaked or mis-evaluated data explains the wrong thing convincingly.

## Process: The Interpretability Gate

### Step 1: Prefer an interpretable model before reaching for a black box

Start with the simplest model that could plausibly work, and only add complexity if it earns its keep:

- Logistic regression, decision trees, and generalized additive models are *intrinsically* interpretable — the coefficients or splits ARE the explanation.
- Compare the interpretable baseline's performance against the complex model honestly. If a logistic regression is within a point or two of a gradient-boosted ensemble, the interpretable model is often the better *deployable* choice in healthcare, even at a small accuracy cost.

```python
# Always fit the interpretable baseline alongside the complex model
from sklearn.linear_model import LogisticRegression
interpretable = LogisticRegression(class_weight="balanced", random_state=42).fit(X_tr, y_tr)
# Compare on the same metric you chose in the metrics skill (e.g. PR-AUC), not accuracy
```

The "interpretability tax" (small accuracy given up for transparency) is frequently worth paying when a human must justify the decision. State the trade-off explicitly rather than defaulting to the most complex model.

### Step 2: Explain global behavior — what drives the model overall

Before explaining single predictions, characterize the model as a whole:

- For linear models: report coefficients (with the direction and, for standardized features, relative magnitude) — but never interpret a coefficient as causal.
- For tree ensembles: report feature importances, and prefer permutation importance over the default impurity-based importance (impurity importance is biased toward high-cardinality features).

```python
from sklearn.inspection import permutation_importance
imp = permutation_importance(model, X_val, y_val, scoring="average_precision",
                             n_repeats=10, random_state=42)
```

Sanity-check the top features against domain knowledge with your domain partner: do the drivers make clinical sense, or is the model leaning on something suspicious (see Step 5)?

### Step 3: Explain individual predictions — what drove THIS patient's score

Clinicians act on individuals, not averages. Provide local explanations:

- **SHAP** (Shapley additive explanations): attributes a prediction to each feature with a sound game-theoretic basis and consistent local+global behavior. The current default for local explanation.
- **LIME**: fits a simple local surrogate around one prediction. Useful, but less stable than SHAP — explanations can vary across runs, so treat it as indicative.

```python
import shap
explainer = shap.Explainer(model, X_tr)
shap_values = explainer(X_val)
# shap.plots.waterfall(shap_values[i])  # why patient i got their score
```

Caveat to state, not hide: SHAP/LIME explain *the model's reasoning*, not clinical truth. They show which features moved the score, not why the disease occurs. Present them as "what the model used," never as a causal claim.

### Step 4: Check calibration before trusting probabilities (link to metrics skill)

If a clinician will read the score as a probability ("30% risk"), it must be calibrated — among patients told 30%, about 30% should have the event. A confident-but-miscalibrated model is dangerous precisely because it looks authoritative. Plot a reliability curve; if needed, apply isotonic or Platt scaling, and note in the report whether calibration was assessed.

### Step 5: Audit for shortcut learning and leakage-by-proxy

Powerful models exploit any correlation, including ones that won't generalize or that encode bias:

- A feature that is a proxy for the outcome (a treatment only given after diagnosis) inflates performance and collapses in deployment.
- A feature that encodes access rather than biology (number of prior visits ~ insurance/wealth, not health) bakes inequity into predictions.
- The classic cautionary pattern: models that keyed on scanner type or text markers rather than the pathology, looking accurate in-sample and failing on new data.

Investigate the top features (Steps 2–3) and ask of each: *would this be available at prediction time, and does it reflect the patient's condition rather than an artifact of how data was collected?* Remove proxies and re-evaluate.

### Step 6: Audit fairness across subgroups

Aggregate metrics hide subgroup harm. Disaggregate the metrics from the evaluation skill across clinically relevant groups (age band, sex, ethnicity where available and lawful), and compare:

```python
for g, idx in groups.items():
    tn, fp, fn, tp = confusion_matrix(y_true[idx], y_pred[idx]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"{g}: recall={recall:.3f}  n={idx.sum()}")
```

Look for large gaps in recall or precision between groups — a model that catches one group's disease far less often is an equity failure even at high overall accuracy. Note the impossibility result honestly: several fairness definitions cannot all hold at once when base rates differ, so name which fairness criterion you optimized and why, rather than claiming the model is simply "fair."

### Step 7: Document an explainability and limitations report

Write `MODEL-CARD.md` (a lightweight model card) recording: intended use and out-of-scope uses, the interpretable baseline vs final model trade-off, global drivers, an example local explanation, calibration status, proxy/shortcut features investigated and removed, subgroup performance with gaps named, and explicit limitations (what the explanations do and don't establish, populations not represented, no causal claims). A model card is the artifact that turns "it works" into "here is what it does, for whom, and where it shouldn't be trusted."

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "It's accurate, that's what matters" | A correct prediction a clinician can't interpret or justify won't be acted on, and shouldn't be. Accuracy buys a hearing, not deployment. |
| "Deep model beats logistic regression by 1%, use the deep one" | In healthcare the interpretability of the simpler model is often worth more than a marginal accuracy gain. State the trade-off; don't default to complexity. |
| "SHAP says feature X causes the outcome" | SHAP explains the model's reasoning, not causation. It shows what moved the score, never why the disease happens. |
| "Overall accuracy is high, so the model is fair" | Aggregate metrics hide subgroup gaps. A model can be accurate overall and systematically miss one group's disease. |
| "The model found prior-visit count is predictive — great feature" | That may encode access to care, not health — a proxy that bakes inequity in. Audit what each top feature actually represents. |
| "Calibration is a nice-to-have" | If a human reads the output as a probability and acts on it, miscalibration is a safety issue, not a nicety. |
| "Explainability is a final polish step" | If you can't explain it, you can't audit it for shortcuts or bias — explainability is how you *find* the problems, not decorate the solution. |

## Red Flags

- A complex black-box model presented with no interpretable baseline for comparison
- Feature importances never sanity-checked against domain knowledge
- A SHAP/LIME plot described in causal language ("X causes Y")
- Probabilities reported and acted on without any calibration check
- A top feature that wouldn't be available at real prediction time (leakage-by-proxy)
- A feature encoding access/wealth/geography treated as a clinical signal
- Only aggregate metrics reported; no subgroup breakdown
- "The model is fair" claimed without naming a fairness definition or showing subgroup numbers
- No model card / limitations documentation accompanying the model

## Verification

- [ ] Interpretable baseline fit and compared to the final model on the chosen metric; trade-off stated
- [ ] Global drivers reported (coefficients or permutation importance) and sanity-checked with domain input
- [ ] At least one local explanation (SHAP) produced and described as model reasoning, not causation
- [ ] Calibration assessed if probabilities are to be acted on; method noted
- [ ] Top features audited for proxies/shortcuts; any leakage-by-proxy removed and model re-evaluated
- [ ] Metrics disaggregated across relevant subgroups; gaps named, not hidden
- [ ] Fairness criterion chosen named explicitly, with acknowledgment that definitions can conflict
- [ ] `MODEL-CARD.md` documents intended use, drivers, calibration, fairness, and limitations (no causal claims)

If any box lacks evidence, the model may be accurate, but it is not yet safe to put in front of a decision-maker.
