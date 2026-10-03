# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** Prithviraj Chowdhury
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Prithviraj-chw/FlyRank-Starter
- **Date:** October 2026

> Deployed paper: https://prithviraj-chw.github.io/FlyRank-Starter/

## 1. Problem framing

The unit of analysis is one page per client, scored monthly. The output is a ranked queue —
each page gets a priority score, an archetype (`confirmed_risk`, `quiet_riser`,
`flagged_lower_urgency`, `stable`), and a human-verb action (`review_before_revert`,
`investigate_quiet_risk`, `verify_then_review`, `monitor_only`). The human action is a content
reviewer deciding which ~50 pages to look at first out of a much larger eligible population.
The cost of a wrong call is bounded by design: nothing in this system auto-edits, auto-reverts,
or auto-publishes — a missed or wrongly-prioritized page costs a week's delay in review, not a
content change. ML helps here because the review budget is small (a human can realistically
check ~50 pages) relative to the eligible population (116,512 pages), so the order of the queue
is the product.

## 2. Data safety

Source: `FlyRank/internship-warehouse` on Hugging Face, `fact_content_daily_performance` joined
with `dim_content` and `dim_clients`, March 2026 window (eligibility) and April 2026 (label
only). Only `client_hash_id` and `content_hash_id` are used anywhere — no raw client or domain
names appear in the notebook, the report, or the deployed paper.

Leakage risks considered and checked (Methodology / Section III of the deployed paper):
- **Label leakage:** neither `decline_label` nor April clicks appear in the feature set.
- **Future leakage:** every feature is March-only, known before the decision point.
- **Downstream leakage:** the rule's outputs (`action`, `reason_code`, `rule_score`) are never
  used as model inputs.
- **Injection test:** a deliberately planted label copy pushes the model's score toward 1.0,
  confirming the leakage audit can actually catch a real leak, not just pass by default.
- IDs are used for grouping only (`GroupKFold` on `client_hash_id`), never as features.

Confirmed: nothing client-identifying appears anywhere in `work/`.

## 3. Baseline

The Week 4 hand-written rule: `snippet_fix` if CTR is below half the position-peer average,
else `content_fix` if engagement rate is below 10% (minimum 10 sessions), else `monitor`. It's a
fair comparison because it's scored on the exact same 28,805 labeled pages and the exact same 5
client-grouped folds as both models — same data, same metric, same split.

**Rule's Precision@50: 0.472** — against a 0.545 base rate. The rule's top-ranked pages decline
*less* often than a random pick; under this honest split it does not beat chance.

## 4. Model / analysis

Logistic regression and random forest, chosen because the first is simple and inspectable
(useful for a reviewer who has to trust the ranking) and the second can catch feature
interactions the first can't. Both trained on the same 8 features: `gsc_impressions`,
`gsc_clicks`, `gsc_avg_position`, `ga4_sessions`, `ga4_engaged_sessions`, `ctr`,
`engagement_rate`, `peer_avg_ctr`, plus position-bucket dummies — all March-only, all known
before the decision point. Features deliberately left out: anything from April (the label
window), and the rule's own outputs.

**Target, in one sentence:** `decline_label = 1` if a page's April clicks fall below 80% of its
March clicks, restricted to pages with at least 5 March clicks so the ratio isn't noise.

## 5. Evaluation

**Split:** 5-fold `GroupKFold` on `client_hash_id` — no client appears in both train and test,
verified with a zero-overlap assertion on every fold. Chosen over a random row split because an
earlier Week 6 audit of this same kind of model showed a random forest scoring 0.90 Precision@10
under a naive random split versus 0.58 once re-evaluated with client grouping — a ~0.3 point
inflation from leakage alone.

**Metrics, model vs. baseline, same split:**

| Method | P@10 | P@50 | P@100 |
|---|---|---|---|
| Rule (baseline) | 0.480 | 0.472 | 0.498 |
| Logistic regression | 0.580 | **0.592** | 0.574 |
| Random forest | 0.600 | 0.560 | 0.566 |

Base rate (random-pick precision): 0.545. Fold-to-fold spread at P@50 (sample SD across 5
folds): rule ±0.14, logistic regression ±0.14, random forest ±0.22.

**Error analysis:** on fold 4 (the weakest), both models underperform the rule — logistic
regression scores 0.42 and random forest 0.26, against the rule's 0.58 on that same fold. A
single average across folds hides this; the model's advantage is real but not uniform across
client groups.

## 6. Interpretation

Standardized logistic regression coefficients, ranked by magnitude (positive = pushes toward
"declining"): `gsc_clicks` −0.266, `gsc_avg_position` −0.191, `ga4_engaged_sessions` +0.119,
`ctr` −0.102, `gsc_impressions` +0.081, `ga4_sessions` −0.081.

In plain words: the three strongest signals (more clicks, better position, higher CTR) all push
a page *away* from the declining label — consistent with the known caveat that lower-traffic
pages have noisier click ratios and are more likely to cross the 80% threshold by chance.

**Negative/surprising result, not smoothed over:** `ga4_sessions` is negative while the closely
related `ga4_engaged_sessions` is positive — a collinearity symptom between two correlated
session-count columns, not evidence that engaged sessions specifically predict decline. The
coefficient *ranking* is more trustworthy than any individual sign.

**Sensitivity check:** removing `gsc_clicks` and `ctr` (both close relatives of the label)
changed random forest's P@50 from 0.560 to 0.556 with matched seeds — a negligible dependence on
these two features in this run.

## 7. Recommendation

The ranked queue, scored on all 116,512 eligible March pages:

| Archetype | Count | Action |
|---|---|---|
| confirmed_risk | 11,210 | `review_before_revert` |
| quiet_riser | 442 | `investigate_quiet_risk` |
| flagged_lower_urgency | 76,547 | `verify_then_review` |
| stable | 28,313 | `monitor_only` |

A FlyRank editor uses this by working down the queue in score order, starting with
`confirmed_risk`, reading the reason code, and checking whether the model's flag matches what
they see on the page before acting — the queue orders review priority, it does not authorize a
change.

**Confidence stated explicitly:** Precision@50 is 0.59, not 0.90 — a meaningful share of flagged
pages will not actually decline. The `HIGH_RISK_CUTOFF` (0.6084) is population-relative and will
shift every cycle. The base rate itself moved from 0.313 (Feb→March) to 0.545 (March→April) —
a proposed ±0.15 alarm threshold would have fired on that shift alone.

**No-go list:** never auto-edit, auto-revert, or auto-publish a content change. Every action
name is a human verb on purpose.

## 8. Reproducibility

Re-run from a fresh clone: open `work/notebooks/capstone.ipynb` in Colab, run the setup cell,
then Runtime → Run all. Every number above comes from that single run.

**Random seeds:** `random_state=0` throughout — both the final deployed model and every
cross-validation fold. **Environment:** Python 3, scikit-learn, pandas, DuckDB
(`requirements.txt` / Colab default stack). **Data access:** `FlyRank/internship-warehouse` on
Hugging Face, read via a gated token (Colab Secrets, never hardcoded).

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> language everywhere — confirmed throughout (Limitations, Interpretation, Recommendation).
> **Metrics vs. base rate:** base rate (0.545) reported next to every Precision@K number.
> No causal claims without an experiment — confirmed (abstract and Limitations state this
> explicitly). No "predicted Google's algorithm" — confirmed, no such claim made anywhere.
> No client-identifying details — confirmed, hashed IDs only. Numbers in this report match a
> fresh re-run — confirmed against `work/figures/capstone_metrics.json`.
