---
name: leaderboard-forensics
description: Interprets public leaderboard feedback against cross-validation — estimates public-LB noise, diagnoses CV–LB gaps, decides which signal to trust, plans how to spend limited daily submissions, and recommends which submissions to select as final. Use when the user reports a public LB score, sees CV and LB move in different directions, has a large CV–LB gap, asks whether to trust CV or LB, plans submissions, worries about other teams jumping, or must choose final submissions before a deadline. Guards against overfitting the public leaderboard.
---

# Leaderboard Forensics

The public leaderboard is a small, noisy sample of the test set. Every decision made by looking at it fits that noise a little more. You extract the real information in it (format bugs, validation mismatch, shift) and protect the team from chasing the rest.

## When to use

- After any submission, to interpret the score.
- CV and public LB disagree, or there is a large constant gap.
- Planning how to spend the remaining submissions.
- Final-selection time.

## Workflow

### 1. Build the submission ledger

For each submission: `sub_id, date, description, cv_mean, cv_sd, public_lb, (private_lb later), selected?`. You need 4 or more points before calling any CV↔LB relationship.

### 2. Estimate the public-LB noise floor

Bootstrap the OOF predictions with sample size = number of public test rows. The std of the metric across about 1000 resamples is the expected LB noise σ_LB.

```python
import numpy as np
def lb_noise(y, oof, metric, n_public, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    s = [metric(y[i], oof[i]) for i in (rng.choice(len(y), n_public, replace=True) for _ in range(n_boot))]
    return np.std(s)
```

LB differences smaller than about 2·σ_LB between two submissions carry almost no information.

### 3. Read the CV–LB relationship

- **Constant offset, same direction:** normal (different sample, maybe mild shift). Keep trusting CV.
- **Same direction for big changes, random for small ones:** normal noise.
- **Systematic disagreement** (CV up, LB down repeatedly, beyond σ_LB): validation mismatch. Diagnose it.

### 4. Diagnose a large or systematic gap, in this order (cheapest and most common first)

1. **Submission bug:** row order or ID misalignment, label mapping, probabilities vs labels, threshold missing, wrong file. Run a full submission audit.
2. **Metric mismatch:** a local metric that differs from the official one (averaging, clipping, label form).
3. **Leaky CV:** optimistic CV from target leakage, group leakage or duplicates across folds.
4. **Train/test shift:** check with a train-vs-test classifier.
5. **Public-subset noise:** the gap is within about 2·σ_LB. Do nothing.

### 5. Decide what to trust

- **Default: trust a well-designed CV.** Use the public LB as a sanity check and as a tie-breaker only when a difference exceeds 2·σ_LB.
- Give the LB more weight only when the public set is large (tens of thousands of rows), the CV↔LB correlation over ≥ 5 submissions is clearly positive, **and** a known shift makes CV less representative.
- If CV and LB still disagree after the diagnosis steps, prefer the choice that is more **robust**: simpler, more diverse, less tuned.

### 6. Spend submissions deliberately

Good uses:
- the first baseline (to check the pipeline and anchor CV↔LB)
- large structural changes (a new feature family, a shift-robust variant, an ensemble)
- testing a hypothesis CV cannot answer

Bad uses: hyperparameter nudges, blend-weight tweaks, threshold tweaks. Keep 2 or more submissions in reserve for the final day.

### 7. Final selection

- **You choose 2 submissions:** (1) the best-CV robust ensemble; (2) a hedge that differs on the main open risk. For example, the shift-robust variant, the best-public candidate *if* it is within noise of the best CV, or a less-tuned blend. Never pick two near-duplicates.
- **You choose 1:** the best-CV robust ensemble, unless the LB contradicts it beyond 2·σ_LB **and** the contradiction has a confirmed explanation.
- **"Last upload counts" rule:** the final upload must be the chosen file. Plan the timing so no experimental file is uploaded after it.
- **Automatic best-public:** your only lever is not uploading public-overfit experiments late. Warn the user.

## Output format

```
## Submission Ledger (table)
## Public-LB noise: σ_LB ≈ ... (n_public = ...)
## CV↔LB reading: offset / correlation / verdict
## Gap Diagnosis: checks done → finding
## Trust Decision: CV | LB | both-with-caveat, and why
## Submission Plan: remaining N → planned uses
## Final Selection: sub A (why), sub B (hedge against ...)
```

## DO NOT

- Do not tune hyperparameters, thresholds or blend weights on the public LB.
- Do not select final submissions by public rank alone.
- Do not probe the leaderboard to infer test labels or public/private membership. It usually breaks the rules and always overfits.
- Do not treat one CV/LB pair, or a difference below 2·σ_LB, as evidence.
- Do not react to other teams' public jumps. They may be public overfitting or a leak the rules will disqualify.
- Do not diagnose shift before ruling out submission bugs and metric mismatch.

## Uncertainty

State σ_LB and how many ledger points support the conclusion. With fewer than 4 points, label the CV↔LB reading "insufficient data" and default to CV. If the public/private split size is unknown, assume the public set is small and noisy.
