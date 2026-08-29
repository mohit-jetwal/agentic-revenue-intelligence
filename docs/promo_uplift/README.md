# Promo uplift — documentation index

The causal capability: **how much incremental sales and profit did a promotion
actually cause?**

Not "sales during the promotion", and not "sales during minus sales before". On
this data that second number overstates incrementality by around 58 percentage
points, and the whole package exists to replace it with one that survives
scrutiny.

## Start here

| Document | Read it for |
|---|---|
| [`business_objective.md`](business_objective.md) | What uplift is, the worked numerical example, and why the obvious calculation is wrong |
| [`causal_methodology.md`](causal_methodology.md) | Potential outcomes, identification, the six estimators, and every measured design decision |
| [`validation.md`](validation.md) | The measured results — synthetic recovery, ground-truth recovery, placebo, sensitivity |
| [`assumptions.md`](assumptions.md) | Every assumption, ranked by how likely it is to be the thing that is wrong |

## Definitions

| Document | Contents |
|---|---|
| [`treatment_definition.md`](treatment_definition.md) | What counts as treatment, the three windows, why the estimand is the ATT |
| [`control_definition.md`](control_definition.md) | Two control pools and how each fails, sufficiency thresholds, matching |
| [`feature_catalog.md`](feature_catalog.md) | Every covariate, every deliberate exclusion, and the anchoring rule |

## Using it

| Document | Contents |
|---|---|
| [`service_contract.md`](service_contract.md) | Request/response shape, the CLI, the exception hierarchy |
| [`business_metrics.md`](business_metrics.md) | Incremental units → revenue → margin → profit → ROI, and the four decisions that change the answer |
| [`failure_modes.md`](failure_modes.md) | Every refusal and warning, plus the bugs the diagnostics have already caught |
| [`databricks_migration.md`](databricks_migration.md) | Local → Databricks design. Design only |
| [`../models/model_cards/promo_uplift_model_card.md`](../models/model_cards/promo_uplift_model_card.md) | Intended use, measured performance, limitations, ethics |
| [`step7_report.md`](step7_report.md) | The step's deliverables against measured results |
| [`interview_guide.md`](interview_guide.md) | The 25 questions, answered |

## The headline numbers

| | |
|---|---|
| Ground-truth recovery, 4,417 real events | expected **+71.3%**, estimated **+72.0%**, error **0.7 pp** |
| Synthetic scenarios recovered | **6 / 6**, intervals cover truth in all six |
| Naive method's bias under confounding | **+57.6 pp** overstated |
| Spurious uplift the naive method invents where the true effect is zero | **+34.9%** |
| Price channel vs mechanic | **+45.6%** vs **+17.7%** |
| Tests | **175** (777 repo-wide) |

## Where the code is

| Path | Purpose |
|---|---|
| `ml/promo_uplift/treatment.py` | Treatment, control and washout roles — the definition everything rests on |
| `ml/promo_uplift/features.py` | Pre-treatment covariates. **The highest-risk module** |
| `ml/promo_uplift/estimators.py` | IPW, AIPW, DR-learner, cross-fitting, clustered SEs |
| `ml/promo_uplift/propensity.py` | `P(T=1\|X)`, overlap, weight stabilisation |
| `ml/promo_uplift/diagnostics.py` | Placebo, sensitivity, the method-aware verdict, ground-truth comparison |
| `ml/promo_uplift/synthetic.py` | Exact-truth generator. **The only thing that can prove correctness** |
| `ml/promo_uplift/pipeline.py` | End-to-end orchestration and the report |
| `app/services/promo_uplift_service.py` | The service seam |
| `app/tools/promo_uplift_tool.py` | The agent-facing contract |

## Commands

```powershell
# Validate — recover known effects
uv run python scripts/estimate_uplift.py --synthetic --all-scenarios
uv run python scripts/estimate_uplift.py --validate-ground-truth --sample-pairs 300

# Estimate and persist
uv run python scripts/estimate_uplift.py --sample-pairs 300 --seed 42
uv run python scripts/estimate_uplift.py --full --no-sensitivity   # every listing

# Query
uv run ari uplift --promotion PR0000123
uv run ari uplift --product P00003 --store S00155
uv run ari uplift-quality
uv run ari uplift-validate

# Test
uv run pytest tests/promo_uplift -v
```

## How this connects to the rest of the platform

```
Baseline (Step 5)      what are normal sales?          ──┐
                                                          ├──→ PROMO UPLIFT (Step 7)
Forecasting (Step 6)   what will sales be?             ──┘         │
                                                                    │
                                        event_impact: incremental profit per promotion
                                                                    │
                                                                    ▼
                                              Trade Promotion Optimization (Step 8)
                                              "where should the next ₹10M go?"
```

**Baseline** supplies one of the six counterfactuals — trained with
`PromotionApproach.EXCLUDE`, it has never seen a promotional row, so its
prediction *is* the no-promotion expectation.

**Forecasting** is orthogonal, not a competitor. It predicts sales *given* a
planned promotion; uplift asks what that promotion *caused*. A forecast is
correct if the number matches what happens; an uplift estimate can match nothing
observable and still be right.

**Step 8** consumes the per-event table and allocates budget against
`incremental_profit`, subject to budget, frequency, margin and inventory
constraints.

**The agent** orchestrates and never calculates. Claude can subtract two averages
— that is precisely the danger, since the arithmetic is trivial and the answer is
wrong by around 58 points. See
[`causal_methodology.md`](causal_methodology.md) and the tool's own description,
which tells the agent not to compute uplift itself.

## Numbering

This capability was specified as **Step 7** and is built as Step 7. The
forecasting docs record an earlier one-step offset between the briefs and the
commit history; see [`../forecasting/README.md`](../forecasting/README.md).
