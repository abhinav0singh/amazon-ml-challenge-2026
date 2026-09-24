---
name: competition-autopsy
description: Runs a structured post-mortem after an ML competition or a competition round ends — public-vs-private shake-up analysis, reconstruction of which decisions helped or hurt judged by the information available at the time, gap analysis against top solutions, a time audit, and 5–7 reusable lessons written into a personal playbook. Use when the private leaderboard is revealed, a round of a multi-round hackathon closes, or the user asks what went wrong, what winners did differently, or wants to update their competition checklist.
---

# Competition Autopsy

You turn one competition's outcome into rules that improve the next one. You judge decisions by the information available when they were made, not by how they turned out, and every lesson must be specific enough to change future behaviour.

## When to use

- The private leaderboard or final results are out.
- A round of a multi-round event closes (for example, an online round before a shortlist or presentation round).
- The user asks "what went wrong?", "what did the winners do?", or wants to update a checklist or playbook.

Not for mid-competition re-planning. That is a planning task.

## Evidence to gather

1. Final public and private rank and score, plus **all** submissions with CV, public and private scores (many platforms reveal private scores for every submission after the close).
2. The experiment log, feature ledger and submission ledger, if they exist, and a rough record of where the hours went.
3. Top solutions: write-ups, shared notebooks, discussion posts, presentation decks. **If none are available, say so. Never invent what winners did.**

## Workflow

### 1. Shake-up analysis

- Rank change public → private, for you and for the top of the board.
- Across your own submissions, compute the correlation of CV with private and of public with private. Which signal was the better guide?
- Was your best-private submission among the ones you selected? If not, measure the gap and find the signal that misled you.
- Were private score differences between your top candidates larger than the noise? Bootstrap estimate if possible.

### 2. Decision reconstruction

List the 5–10 key decisions: CV design, leakage handling, feature families, model mix, tuning effort, ensemble method, threshold, final selection. For each one:

| decision | evidence at the time | alternative considered | outcome | verdict |

Verdicts:
- **Good call**
- **Good call, bad luck** (sound reasoning, noise went against it)
- **Lucky** (poor reasoning, good outcome; do not repeat)
- **Mistake** (the information was available and not used)

### 3. Gap to the winners

Categorise what the top solutions did that you didn't: data insight or leak, validation, features, models, ensembling, post-processing, external or original data, compute. Estimate which category explains most of the score gap. For each, extract the **transferable principle**, not the recipe. Example: "they aggregated by the hidden entity key" becomes the principle "search for implicit entity IDs formed by combinations of columns."

### 4. Time audit

Hours per phase vs the value each produced. Name the biggest sink of time with little score gain, and the highest-return hour.

### 5. Lessons (5–7 at most)

Each lesson is a rule with a trigger, an action and evidence:

> **When** [situation], **do** [action], **because** [evidence from this competition].

Tag each one Keep / Start / Stop. Drop any lesson that would apply to every competition equally ("do more feature engineering"). Those are not lessons.

### 6. Playbook update

Produce the exact lines to append to `playbook.md`, grouped by phase (validation, features, ensembling, submission, time management). Mark lessons that repeat earlier ones; repetition means the rule needs to become a checklist item.

## Output format

```
## Result: public #.. → private #.. (Δ ..) | score ..
## Shake-up: CV↔private r = .., public↔private r = .. → trusted signal was ..
## Selection Check: best private sub = .., selected? .., cost ..
## Decision Review (table with verdicts)
## Gap to Winners (category | what they did | principle | est. impact) — or "no write-ups available"
## Time Audit
## Lessons (Keep / Start / Stop)
## Playbook Additions (ready to paste)
```

## DO NOT

- Do not draw conclusions from private differences smaller than the noise ("B beat A by 0.0004 on private, so B's idea was better").
- Do not judge decisions by outcome alone (outcome bias).
- Do not write generic lessons, or more than 7.
- Do not fabricate winner solutions or their scores.
- Do not assign blame to teammates. Analyse decisions and processes.

## Uncertainty

When the evidence can't separate "bad luck" from "mistake" (for example, only 2 submissions have private scores), say so and mark the lesson **tentative** until another competition confirms it.
