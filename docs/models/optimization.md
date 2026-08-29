# Optimisation and scenario simulation

Three capabilities, one step, because they share a shape: each consumes the
estimates the earlier steps produced and returns a **decision** rather than a
measurement — and so inherits every assumption underneath it.

| Capability | Question | Consumes |
|---|---|---|
| Trade promo allocation | Where should the next ₹10M go? | Step 7's per-event incremental profit |
| Price optimisation | What price should we set? | Step 8's own and cross elasticity |
| Scenario simulation | What if we cut price *and* promote? | Both, plus the sales base |

---

## Trade promotion allocation

### Diminishing returns are the whole problem

The generator builds promotional response as `a·(1 − e^(−b·d))` — the tenth point
of discount buys far less than the second. A linear program over a constant
ROI-per-rupee ignores that and pours the entire budget into the single
highest-ROI cell, which is both wrong and obviously wrong to any category
manager.

So each cell's response is approximated as a **piecewise-linear concave** curve
(five segments) and solved with OR-Tools GLOP. Concavity is what makes the budget
spread: each successive segment returns less than the last.

**Measured**: three cells with profit rates 3.0, 2.0 and 0.5 and a ₹150 budget.
A linear model puts all 150 into the first. This allocates **80 / 70 / 0** —
and stops entirely at 600 of a 1,000 budget, because beyond that no cell has a
positive marginal return left.

### Infeasibility is a result, not an error

"Your minimum regional spends already exceed the budget" is a genuine business
finding. The solver reports the conflict and names it:

```
status = infeasible
  ! dimension minimum spends total 200, which already exceeds the budget of 50
```

### Two guards worth naming

**No cell is funded beyond twice its observed spend.** Extrapolating a saturating
curve past the range any promotion has actually visited produces a confident
recommendation nobody should act on.

**A constraint that matches no candidate is reported, never silently dropped.**
This was a real bug, found by running the tool rather than by reading it: a
caller asked for ₹900,000 minimum in the North, the event table carried no
`region` column, so every candidate's region was `None`, nothing matched, and the
constraint was quietly ignored — *and then reported as binding* because the
region's spend was zero. The caller would have been misled twice. Step 7's event
table now carries `region`, `category` and `channel`, and an unmatched constraint
raises a warning that downgrades the tool result to `partial`.

### The caveat that governs interpretation

Step 7 measured promotional spend in the generated data at roughly **20× the
achievable margin** at product-store grain, so absolute ROI is not interpretable.
The allocation is validated on **ranking** and **constraint satisfaction** — does
it prefer the genuinely better promotions, does it respect its bounds — not on
the profit number it reports.

---

## Price optimisation

### Why a range, not a price

For constant elasticity the unconstrained optimum has a closed form:

```
p* = c · e / (1 + e)
```

**The closed form is deliberately not what ships**, for three reasons.

*Constraints.* Margin floors and price-change caps are the normal case, and the
unconstrained optimum usually violates one.

*False precision.* An optimum computed from an elasticity whose interval spans
−1.8 to −2.6 is a point estimate of a point estimate. The honest output is the
range over which profit is near-flat, and a category manager will act on
"103–106 are equivalent" where they would rightly distrust "set it to 104.37".

*Cannibalisation.* Optimising a product alone reliably recommends a rise that
moves volume to its own category neighbour and books a phantom gain.

So a grid of candidate prices is evaluated, each scored on **portfolio profit net
of cannibalisation**, and the recommendation is the best candidate plus every
candidate within 2% of it — a tolerance chosen against the elasticity's own
uncertainty rather than as a round number.

**Verified against theory**: at `e = −2`, `c = 60`, the optimiser returns
**120.00**, exactly the closed-form `60 × (−2) / (−1)`.

### The grid-edge guard

The first version used a ±15% grid. The optimum for `e = −2` is +20%, so **every
recommendation pinned to the grid's own edge** — the answer was an artifact of
the grid, not of the demand curve, and cannibalisation could not change it
because both cases were already pinned.

Widened to ±30%, and `recommend()` now warns when the optimum still lands on the
boundary: *"treat this as 'move at least this far', not as the best price"*.

With the wider grid, a substitute in the portfolio moves the optimum from **120
to 130** — raising this product's price sends volume to something you also own,
so the optimal rise is larger. That is the cannibalisation effect finally
visible.

---

## Scenario simulation

Composes the other models and fits nothing of its own.

### Levers compose in log space

Effects accumulate as log terms and are exponentiated once, matching the demand
equation they are projected through. Applying them multiplicatively one at a time
would give the same answer for a single lever and a different one for several,
because the order would start to matter. Tested: forward and reverse lever order
give identical results.

### Confidence is the weakest link, never an average

| Component | Confidence | Why |
|---|---|---|
| price_elasticity | 0.85 | recovered known elasticities at r=0.99 |
| promo_uplift | 0.80 | recovered ground truth to 0.7pp on 4,417 events |
| baseline_sales | 0.75 | sits at 1.15× the irreducible noise floor |
| cross_price_elasticity | 0.60 | correct, but a much smaller effective sample per pair |
| competitor_price | 0.40 | an assumption, not an estimate — nothing says how a rival reacts |

A scenario chaining a price move and a competitor response reports **0.40**, not
the 0.62 an average would give. A projection is only as trustworthy as its
shakiest input, and averaging describes a projection nobody made.

### Ranges come from the inputs' own intervals

The projection is recomputed at each end of the elasticity and uplift intervals.
Crude compared with full error propagation, and honest about what it is: a band
showing how far the answer moves when the inputs move across the range that was
actually measured. Where no interval exists, the result **says so** rather than
presenting a point as certain.

**Risk was recalibrated after a measurement.** An 87k–227k profit band around a
156k central estimate was being reported as *low* risk under a 2× spread
threshold. A range as wide as the central estimate means the projection
establishes the effect's sign but not its size — the threshold is now 1×.

### Unmodelled levers are reported

An `inventory` lever is not modelled. The projection says so in its warnings
rather than silently returning a number that excludes it.

---

## Usage

```powershell
uv run pytest tests/optimization -v
```

```python
from app.services.container import Container
svc = Container().optimization_service

svc.allocate(total_budget=10_000_000)
svc.optimize_price(product_id="P00003", min_margin_pct=0.25)
svc.simulate(levers=[...], product_ids=["P00003"], horizon_days=30)

# Several options against ONE baseline. Computing them independently would let
# baseline drift masquerade as a difference between the options.
svc.compare_scenarios(scenarios=[[...], [...]], product_ids=["P00003"])
```

## Files

```
ml/trade_promo_optimization/optimizer.py   concave allocation, OR-Tools GLOP
ml/trade_promo_optimization/model.py       wired to Step 7's event table
ml/price_optimization/optimizer.py         candidate grid, portfolio scoring
ml/price_optimization/model.py             wired to Step 8's elasticity
ml/scenario/engine.py                      log-space lever composition
ml/scenario/model.py                       composes the fitted models
app/services/optimization_service.py       one service, three capabilities
app/tools/optimization_tools.py            three agent-facing tools
tests/optimization/                        38 tests
```
