---
name: clinical-model-monitoring-and-drift
description: Guides agents through monitoring a deployed health-adjacent model for data drift, performance decay, and silent failure, and through deciding when to alert, retrain, or roll back. Use when a model is in or near production, when planning what to log and watch after deployment, when performance complaints arrive, or when designing the human-in-the-loop safety net. Use even when a model launched accurate — in healthcare, accuracy at launch is a snapshot, and a model that degrades unnoticed causes harm precisely because it still looks confident.
---

# Clinical Model Monitoring and Drift

## Overview

Post-deployment safety for models that inform health decisions. Training and evaluation prove a model worked on yesterday's data; they say nothing about whether it still works next quarter. Patient populations shift, lab equipment changes, coding practices update, a pandemic rewrites baseline rates — and a model keeps emitting confident scores throughout, with no error message when those scores stop being right. This skill is the gate that runs continuously after deployment: it defines what to log, what to watch, what thresholds trigger action, and how a human stays in the loop so silent decay becomes a caught incident.

Core principle: **a deployed model is a perishable asset, not a finished artifact.** The dangerous failure in production is not the model that crashes — it is the one that keeps running while quietly becoming wrong.

## When to Use

- A model is deployed, piloted, or about to be put in front of real decisions
- Planning observability: what to log, what dashboards and alerts to build
- A performance complaint or unexpected outcome arrives from the field
- Designing the human-in-the-loop / override / escalation path
- Deciding whether and when to retrain, recalibrate, or roll back
- Periodic scheduled review of a live model's health

**NOT for:** the pre-deployment gates — privacy, quality, honest metrics, and interpretability each come first. Monitoring a model that was never properly evaluated just tracks the wrong number over time.

## Process: The Monitoring Gate

### Step 1: Decide what to log before launch, not after

You cannot monitor what you did not record. Before deployment, instrument the system to log, for every prediction: input feature values (subject to the privacy skill — store de-identified or hashed-as-surrogate, never raw identifiers), the model's score and the decision at the chosen threshold, the model version, a timestamp, and — when it later becomes available — the **outcome / ground-truth label**.

The hardest and most important part in healthcare is **label latency**: the truth (did the patient actually deteriorate, get readmitted, have the disease?) often arrives weeks or months later, or never. Plan how outcomes will be captured and joined back to predictions, because without labels you can only watch inputs, not correctness.

### Step 2: Monitor data drift (the inputs)

Data drift is a change in the distribution of the inputs the model receives versus what it was trained on. It is the earliest available warning because it needs no outcome labels.

```python
# Population Stability Index per feature: train distribution vs recent window
import numpy as np
def psi(expected, actual, bins=10):
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual,   cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return np.sum((a - e) * np.log(a / e))
# Rule of thumb: PSI < 0.1 stable; 0.1-0.25 moderate shift; > 0.25 significant
```

Track per-feature drift (PSI, or a Kolmogorov-Smirnov test for continuous features, chi-square for categorical) and the drift of the model's own output scores. Rising input drift is a signal to look harder, not an automatic failure — investigate the cause before reacting.

### Step 3: Distinguish the kinds of drift, because the fix differs

- **Data / covariate drift:** inputs change, the input-to-outcome relationship holds (a clinic starts serving an older catchment). May need recalibration or retraining on recent data.
- **Concept drift:** the relationship itself changes (a new treatment alters who deteriorates; a coding standard redefines a diagnosis). The old patterns are now wrong — retraining on fresh labels is usually required.
- **Upstream / pipeline drift:** a feed changes units, a field starts arriving null, a sensor is replaced. This is a *data-quality* failure in production (see the quality skill) masquerading as model decay — fix the pipeline, do not retrain on corrupted inputs.

Name the kind before choosing the response. Retraining a model whose real problem is a broken upstream feed bakes the corruption in.

### Step 4: Monitor performance when labels arrive

Once outcomes are available, recompute the metrics chosen in the evaluation skill (recall, precision, PR-AUC) on recent data and compare to the launch baseline — disaggregated across the same subgroups from the fairness audit, because decay often hits one group first. A model can hold steady overall while collapsing for a subpopulation.

Watch **calibration drift** specifically: a model can keep ranking well (stable AUC) while its probabilities drift out of calibration, so a "20% risk" no longer means 20%. If clinicians act on the probability, this is a safety regression even with stable discrimination.

### Step 5: Set thresholds and an action ladder in advance

Decide, before anything fires, what each signal triggers — so a 3am alert meets a plan, not improvisation:

| Signal | Example trigger | Pre-agreed action |
|---|---|---|
| Input drift | PSI > 0.25 on a key feature | Investigate cause; notify owner; no auto-change |
| Performance drop | Recall falls below an agreed floor | Escalate to clinical owner; consider rollback |
| Calibration drift | Reliability curve off beyond tolerance | Recalibrate; flag scores as provisional meanwhile |
| Pipeline anomaly | Feature suddenly all-null / unit shift | Halt automated use; fix feed before resuming |
| Subgroup gap widens | One group's recall drops disproportionately | Pause for that group; equity review |

Crucially, **automated detection should rarely trigger automated clinical change.** In healthcare the alert routes to a human owner who decides — the loop is detect → notify → human judgement → act, not detect → auto-retrain.

### Step 6: Plan retraining and rollback as deliberate, reversible events

- **Rollback first:** the fastest safe response to a bad model is reverting to the last known-good version. Keep versions and their validation records so rollback is one step, not an archaeology project.
- **Retraining is a release, not a reflex:** a retrained model goes through the same gates as the original — quality, evaluation, interpretability, fairness — before it replaces the incumbent. Never hot-swap an unvalidated model just because it trained on newer data.
- Beware **feedback loops:** if the model's own decisions shaped the data you now retrain on (it flagged patients who then got treated, changing their outcomes), naive retraining learns its own past behavior. Note and account for this.

### Step 7: Keep the human in the loop and document the monitoring plan

Write `MONITORING.md` recording: what is logged and how outcomes are captured (and their latency), which drift and performance metrics run on what schedule, the threshold/action ladder, who the clinical owner and on-call human are, the rollback procedure and current known-good version, and a log of incidents and responses. A monitoring plan that lives only in someone's head is not a plan; the document is what lets the model be operated safely by people who did not build it.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "It validated well, so it's done" | Validation is a snapshot of the past. Populations and practices drift; a static model decays against a moving world. |
| "No errors in the logs, so it's healthy" | Silent decay throws no exceptions. The model keeps emitting confident, increasingly wrong scores. Monitor distributions and outcomes, not just crashes. |
| "Accuracy is the thing to watch" | Without fresh labels you often can't measure accuracy at all — so watch input drift and score drift, which are available immediately, and add performance once labels arrive. |
| "Drift detected — retrain automatically" | Auto-retraining can bake in upstream corruption or a feedback loop, and an unvalidated retrain is an unreviewed model. Detection alerts a human; retraining is a validated release. |
| "Overall metrics are stable, we're fine" | Decay often hits one subgroup first. Monitor disaggregated, or you'll discover the harm after it's done. |
| "Calibration was fine at launch" | Calibration drifts independently of ranking. A still-discriminating model can become miscalibrated and unsafe to act on as a probability. |
| "We can just swap in the newer model" | Hot-swapping an unvalidated model bypasses every safety gate. Retrains earn deployment by passing the same gates as the original. |

## Red Flags

- A model in production with no logging of inputs, scores, versions, or outcomes
- Monitoring that only checks uptime/errors, not data and prediction distributions
- No plan for how ground-truth labels will be captured or their latency handled
- Automated retraining or auto-deployment with no human validation step
- Only aggregate performance tracked; no subgroup or calibration monitoring
- No defined thresholds or action ladder — alerts with no pre-agreed response
- No versioning or rollback path to a known-good model
- Retraining on data shaped by the model's own past decisions, unaccounted for

## Verification

- [ ] Pre-launch logging covers inputs (de-identified), scores, decisions, model version, timestamp, and a path to capture outcomes
- [ ] Label latency understood and a mechanism to join outcomes back to predictions is planned
- [ ] Input/data drift monitored (PSI or distributional tests) on key features and on output scores
- [ ] Drift type (covariate / concept / pipeline) diagnosed before any corrective action
- [ ] Performance recomputed on fresh labels, disaggregated by subgroup; calibration drift watched
- [ ] Threshold + action ladder defined in advance; automated detection routes to a human, not to automated clinical change
- [ ] Versioning and a one-step rollback to a known-good model exist; retrains pass the full gate set before replacing the incumbent
- [ ] Feedback-loop risk considered when retraining
- [ ] `MONITORING.md` documents logging, metrics, schedule, thresholds, owners, rollback, and an incident log

If any box lacks evidence, the model is running unsupervised — and an unsupervised health model that looks confident while drifting is more dangerous than no model at all.
