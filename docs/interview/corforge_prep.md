# Corforge — AI Lead / Principal AI Lead

Interview preparation. Read Part 1 before anything else.

---

# PART 1 — Positioning: what to claim and what not to

## The situation

**Stage 1 is complete — all 15 steps.** The roadmap was compressed from 23 steps
to 15 partway through; the compression merged analytical capabilities and
prioritised the agent layer, and it is worth saying so if asked, because
"I re-scoped when the pace made the deadline unrealistic" is a Principal-level
answer.

You can now demonstrate the thing the CV describes. It runs.

| Built, tested, measured | Honestly still absent |
|---|---|
| Synthetic data platform with hidden ground truth | Databricks (design documents only) |
| Point-in-time-correct feature layer | Agentic RAG (`VectorStore` is an interface, no corpus) |
| Baseline Sales · Demand Forecasting (conformal intervals) | Azure anything |
| **Promo Uplift (causal: AIPW, DR-learner, cross-fitting)** | Neo4j, MCP, LangSmith, OpenTelemetry |
| Own- and cross-price elasticity (panel FE, 2SLS diagnostics) | Async job queue (investigations run synchronously) |
| Trade-promo allocation · price optimisation · scenario engine | Unity Catalog model listing (`GET /models` 501s there) |
| Claude provider + deterministic offline stub | |
| **LangGraph plan/act/observe/critique loop** | |
| Critic, bounded re-planning, HITL interrupt | |
| Output validation: every numeral checked against tool results | |
| Golden-set agent evaluation with a committed baseline | |
| FastAPI (all endpoints live), SQLite app state, Streamlit UI | |
| **1,013 tests passing**; ruff, mypy, bandit clean | |

Dockerfile and compose are **written but never built** — Docker is not installed
on the development machine. The lockfile is verified to resolve against the base
image and the compose file parses, but say "written, unproven" if it comes up.
Claiming a green build you have not seen is the one thing on this page that
would actually cost you the room.

Six tools are registered. The prompt files contain real prompts. The 501 stubs
are gone except one, and that one is honest.

## Where the story has moved

The old positioning was "I built the tool layer first, deliberately". That is
still true and still the right sequencing argument — but it is now a story about
*why the finished thing is shaped this way*, not an explanation for something
missing.

The differentiators, in order:

**The causal work is genuinely rare.** Most people applying for AI Lead roles
have RAG chatbots. Very few can explain why conditioning on discount would have
reported +17.7% instead of +71.3%, or why their confidence intervals were three
times too narrow until they clustered the standard errors.

**The hallucination control is architectural, not prompted.** Every numeral in
every recommendation is extracted and checked against the tool results in state.
A prompt instruction with no check behind it is a hope.

**The evaluation has a floor.** The golden set is scored against a deliberately
weak keyword planner, and the honest finding is that **keyword matching ties a
language model on tool selection** for well-posed questions. The LLM earns its
place on abstention and on the trap case. Almost nobody brings a benchmark that
makes their own system look unnecessary in one column.

**The system knows what it cannot answer.** Nine of twenty golden questions are
unanswerable with the registered tools, and they are scored on whether the agent
*declines*. That is the dimension most benchmarks omit.

## The sentence to use

> "It's a working agentic platform: Claude and LangGraph do the reasoning —
> what to investigate, in what order, when the evidence is sufficient — and
> deterministic models produce every number. The LLM never calculates. I built
> the tool layer and its contracts first, on purpose: an agent that calls a tool
> returning a wrong number just launders that error into a business
> recommendation with a confident tone on top."

Then, if asked what is missing:

> "It's Stage 1 — local, single-process, synchronous. Stage 2 is the Databricks
> migration, and that's designed rather than built. And I'd tell you the golden
> set says a keyword planner matches the LLM on tool selection for well-posed
> questions — the model earns its place on knowing when to decline, not on
> routing."

That second answer is the one that lands. Volunteering a result that
undercuts your own system reads as someone who measures things.

## What you must not do

- Do not claim Azure anything. The project uses Anthropic Claude and is
  Databricks-targeted. There is no Azure in it.
- Do not claim Neo4j, MCP, LangSmith or OpenTelemetry. None are present.
- Do not claim agentic RAG. `VectorStore` is an interface with no document
  corpus behind it — it would be scaffolding for its own sake.
- Do not quote a Claude golden-set score. Only the stub baseline is committed;
  a Claude run costs money and has not been recorded.
- Do not say the Docker image has been run. It is written and the lockfile
  resolves for its base image, but Docker is not installed on the dev machine,
  so it has never been built. Say that if asked.

For the first two, Part 4 Section F gives you a real answer.

---

# PART 2 — The three-minute pitch

Memorise the shape, not the words.

> **The problem.** A CPG business runs about ₹100M a year of trade promotions.
> Nobody can say which ones worked. The number the industry reports — sales
> during the promotion minus sales before it — is wrong by a factor of two,
> because promotions are scheduled into rising demand and because shoppers buy
> ahead and then buy less afterwards.
>
> **The idea.** An agentic platform where Claude and LangGraph do the *reasoning*
> — what to investigate, in what order, when the evidence is sufficient — and
> deterministic models produce *every number*. The LLM never calculates. That
> separation is the core architectural commitment.
>
> **What I built.** A synthetic CPG dataset whose generator records the true
> relationships in a directory the application physically cannot read, so every
> model is scored against known truth. A point-in-time-correct feature layer.
> Six analytical models behind six agent tools. Then the agent layer on top: a
> LangGraph plan/act/observe loop, a separate Critic that can send it back to
> re-plan, and a recommendation step where every number is verified.
>
> **The hardest one.** Promo uplift is causal, not predictive. You're estimating
> what *would* have happened without the promotion — a quantity that is absent
> from every dataset that will ever exist. So you can't validate it by holding
> out data. I built a generator where the effect is known exactly, and the
> estimator recovers it in six of six scenarios. On the real dataset, validated
> against the generator's own recorded parameters across 4,417 promotion events:
> expected +71.3%, estimated +72.0%. Error of 0.7 percentage points.
>
> **The control I'd point at.** Every numeral in a final recommendation is
> extracted and matched against the tool results in state. Not a prompt asking
> the model to behave — a check. It reports rather than blocks, because
> legitimate arithmetic over sourced values exists and a check that fires on it
> would get switched off.
>
> **How I know it works.** A golden set of twenty questions derived from the
> scenarios the generator injected, so the right answer exists independently of
> the thing being graded. Nine of them the platform *cannot* answer, and those
> are scored on whether it declines.

Stop there. Let them pick the thread.

---

# PART 3 — End-to-end walkthrough

## The architecture in one picture

```
┌──────────────────────────────────────────────────────────┐
│  INTERFACE            FastAPI · Streamlit · CLI           │
│                       one InvestigationService behind all │
├──────────────────────────────────────────────────────────┤
│  GUARDRAILS           budget · output validation          │
│                       HITL interrupt · approval threshold │
├──────────────────────────────────────────────────────────┤
│  AGENT LAYER          Supervisor · Critic · Recommendation│
│                       LangGraph: plan→act→observe→critique│
│                       └─ bounded re-plan cycle            │
├──────────────────────────────────────────────────────────┤
│  TOOL CONTRACT        AnalyticalTool → ToolResult         │
│                       status · provenance · confidence    │
│                       · assumptions · warnings · trace_id │
├──────────────────────────────────────────────────────────┤
│  SERVICE LAYER        Baseline · Forecasting · Uplift     │
│                       · Elasticity · Optimization         │
├──────────────────────────────────────────────────────────┤
│  MODEL LAYER          baseline · forecasting · uplift     │
│                       · elasticity · optimisation ·       │
│                       scenario                            │
├──────────────────────────────────────────────────────────┤
│  FEATURE LAYER        FeatureEngineer, availability       │
│                       classes, PointInTimeView            │
├──────────────────────────────────────────────────────────┤
│  DATA ACCESS          DataRepository (ABC)                │
│                       DuckDB local │ Databricks prod      │
├──────────────────────────────────────────────────────────┤
│  STORAGE              Parquet gold tables (analytics)     │
│                       SQLite (investigations, traces)     │
└──────────────────────────────────────────────────────────┘
```

**The one-sentence version of the arrow direction:** reasoning flows down,
numbers flow up, and nothing in the top three layers is allowed to compute a
figure the bottom five did not produce.

## The three seams (Step 1)

These are the architectural decisions everything else depends on. Lead with
these — they show platform thinking, which is what a Principal role is about.

**1. `DataRepository`** — an abstract base class defining every read the platform
performs. Business logic receives a repository and never asks which
implementation it got. DuckDB locally, Databricks SQL in production. This is what
makes the Databricks migration a redeployment rather than a rewrite.

**2. The `ToolResult` envelope** — every analytical capability returns the same
shape:

| Field | Purpose |
|---|---|
| `status` | success / partial / error / invalid_input / timeout |
| `model_name`, `model_version`, `dataset_version` | provenance — which artifact produced this number |
| `result` | the payload, as plain JSON |
| `confidence` | **measured, or absent.** Never invented |
| `assumptions` | what the caller must know to use the number responsibly |
| `warnings` | caveats; a non-empty list downgrades status to `partial` |
| `error` | code, message, and a `recoverable` flag for agent re-planning |
| `trace_id` | ties the result to its log lines |

`AnalyticalTool.run()` is marked `@final`, so a subclass cannot bypass
validation, timing, tracing or error wrapping. It implements only `_execute`.

**3. The DI container** — one class with cached properties, keyed on
`APP__ENVIRONMENT`. Components are built lazily, so constructing the container
never requires a trained model or an API key.

## The dataset (Step 2) — and why it is the cleverest part

A synthetic CPG panel: ~300 products × 200 stores × 3 years. Log-additive demand:

```
log λ = log base + season + day-of-week + holiday + trend
      + β_own·log(price/ref)
      + Σ β_cross·log(price_j/ref_j)
      + promo_lift + pull_forward
      + γ·log(competitor_price/ref)
      + noise

latent_units   ~ NegativeBinomial(exp(log λ))
observed_units = min(latent_units, inventory_available)
```

**The key move**: every relationship the platform later claims to *estimate* is
drawn first, recorded to a `ground_truth/` directory, and only then used to
generate sales. `LocalDataRepository` has **no method that can reach it** — the
separation is structural, not a convention.

That inverts the usual synthetic-data problem. Instead of "the model produced
−1.38, is that right?" with no way to answer, you compare −1.38 against a known
−1.42 and report the error.

It also gives you an **irreducible noise floor**: comparing `mean_demand` to
`latent_units` gives 35.0% WMAPE. A model scoring 40% is at 1.15× the floor, not
"60% accurate". That reframes every accuracy conversation.

## Deliberate realism in the generator

Four properties, each of which makes a later step honest:

- **Diminishing returns** on discount depth — otherwise the optimiser pours the
  whole budget into the deepest discount.
- **Pull-forward** — demand dips after a promotion. This is *why* naive uplift
  overstates.
- **Targeted promotions** — timing is drawn with weights `exp(0.40 × 2 ×
  seasonal)`, so promotions land on seasonal peaks. This is the confounder.
- **Endogenous stockouts** — demand outruns the reorder policy, so stockouts
  correlate with high demand.

## Step 4 — Baseline Sales

"What would normal sales be?" Used by uplift, root-cause and the scenario engine.

Two approaches built and compared: **Approach C** (train only on non-promotional
rows — the model has literally never seen a promotion, so its prediction *is* the
no-promotion expectation) and **Approach B** (train on everything with promotion
features, then predict with them zeroed).

**Result**: LightGBM/exclude at 40.4% WMAPE = 1.15× the noise floor.

**The honest bit worth telling**: Approach C won by 0.2 points — far too narrow
to call a general result. And the selected model over-predicts by +6.7%, which
means uplift measured against it is understated by roughly that much. The doc
flags the tension rather than resolving it quietly, and hands the decision to
whoever owns the uplift numbers. Step 7 then owns it and corrects for it.

## Steps 5–6 — Demand Forecasting

**Direct multi-step, one global model, horizon step `h` as a feature.** A training
row is `(origin t, horizon h, target = units at t+h)`.

Rejected alternatives, with the dispositive reason:

| Alternative | Why not |
|---|---|
| Recursive one-step-ahead | **Feature-distribution collapse** — feeding conditional means forward drives rolling std to zero, so by h=30 the model sees inputs it never saw in training |
| Per-horizon cumulative models | Can't produce a daily path; nested totals mutually incoherent |
| Four direct daily models | 4× compute, less data each, no gain once `h` is a feature |

**Results**: XGBoost 43.8% WMAPE (1.25× noise floor), +12.6pp Forecast Value
Added over seasonal naive at every horizon bucket. Error falls 4.5× from SKU
(43.6%) to total (9.6%).

**Conformal prediction intervals**, calibrated per horizon bucket on a dedicated
fold, with achieved coverage measured and reported whatever it is.

Two war stories worth having ready:

- **XGBoost came in at 82.9% WMAPE with +58% bias.** Cause: `min_child_weight`
  sums *Hessians*, and under `count:poisson` the Hessian ≈ μ, so it scales with
  target level. LightGBM's `min_child_samples` counts rows. Scaling to `50 ×
  mean(y)` fixed it: 46.3% at −1.8% bias.
- **Supply features taught the model censoring.** `closing_inventory_lag_1` was
  the #1 feature and the model recovered only 0.30 of true stockout demand —
  it had learned to read low stock as low demand. Excluding supply features:
  0.30 → 0.68 recovery, accuracy essentially unchanged.

## Step 7 — Promo Uplift (the centrepiece)

**This is your best material. Spend the most time here.**

### Why it is a different kind of problem

| | Question | Validatable by holding out data? |
|---|---|---|
| Forecasting | What **will** sales be? | Yes — the answer arrives |
| Uplift | What **would** sales have been without the promotion? | **No** |

The counterfactual is missing from every dataset. Hold out any share of rows and
it is still missing from the held-out part. **A model can predict the outcome
perfectly and be wrong about the effect by any margin.**

### The worked example (have this ready — it lands)

A store sells 1,000 units/week normally. During the promotion: 1,600. The four
weeks before averaged 1,150, because a seasonal peak was building.

| Quantity | Units |
|---|---|
| Observed | 1,600 |
| **Naive (during − before)** | **+450** ← wrong |
| Baseline counterfactual | 1,320 |
| **Gross incremental** | **+280** |
| Pull-forward payback | −90 |
| **Net incremental** | **+190** |

**The naive number is 2.4× the truth.**

### Treatment is the whole event, not the flag

The single most important modelling decision. A promotion moves demand two ways:

```
τ_log = a·(1 − e^(−b·d))    +    β_own · log(1 − d)
        └─── mechanic ───┘        └── price cut ──┘
```

Measured across 4,417 events: **mechanic +17.7%, price channel +45.6%**. The
price cut is **2.6× the mechanic**.

So `discount_percentage` is a **mediator**, not a confounder. Conditioning on it —
which is the natural thing to do, since discount looks obviously relevant —
blocks the larger causal path and reports +17.7% as the whole effect. A number
that looks entirely plausible and is wrong by a factor of four.

### The six estimators

| Method | What must be true | Role |
|---|---|---|
| Naive | nothing — it's wrong | The foil, kept and reported |
| Baseline counterfactual | Step 4's baseline is unbiased | Reuses the artifact |
| DiD | parallel trends | **With a test that can reject it** |
| IPW | propensity model correct | Fragile at extremes |
| **AIPW** | **either** nuisance model correct | **The headline** |
| DR-learner | as AIPW + a CATE model | Segment ranking |

**AIPW in one line:**

```
τ̂ = (1/n₁) Σ [ Tᵢ(Yᵢ − μ̂₀(Xᵢ)) − (1−Tᵢ)·(êᵢ/(1−êᵢ))·(Yᵢ − μ̂₀(Xᵢ)) ]
```

Doubly robust: consistent if **either** the outcome model or the propensity model
is right — not both.

### The validation results

| Scenario | True | Naive | AIPW | CI covers truth |
|---|---|---|---|---|
| positive | +63.6% | +53.5% | **+65.9%** | yes |
| negative | −9.4% | −14.3% | **−5.9%** | yes |
| null | 0.0% | −5.6% | **+3.8%** | yes |
| confounded | +65.9% | **+123.5%** | **+60.9%** | yes |
| confounded_null | 0.0% | **+34.9%** | **+0.2%** | yes |
| heterogeneous | +67.7% | **+126.1%** | **+63.1%** | yes |

**6/6 recovered.** Then on the real dataset: expected **+71.3%** from the
generator's recorded parameters, estimated **+72.0%** across 4,417 events —
**0.7pp error**.

The row to point at is **`confounded_null`**: promotions targeted at exactly the
days that would have sold well anyway, doing nothing. Naive finds **+34.9%** of
pure fiction. AIPW returns **+0.2%**.

### The three debugging stories (rehearse these)

**1. −424% against a true +65%.** I cross-fitted on contiguous *date* blocks —
the intuitive choice, since it's what a forecasting split does. Every fold was
then predicted by a model trained on *other* periods, so the linear time
covariate was extrapolated; propensity scores hit the clip boundaries and control
weights summed to **43×** the treated count. Fixed by holding out whole listings.
The permanent guard came out of it: since `E[(1−T)·e/(1−e)] = P(T=1)`, control
weights must sum to about the treated count, and a ratio outside [0.7, 1.4] now
raises a warning.

**2. The intervals were wrong, not the estimates.** With i.i.d. standard errors,
coverage of known truth was **4/6** while every point estimate was within 2–5
points. Rows within a listing are strongly serially correlated, so the effective
sample size is closer to the number of *listings* than the number of rows.
Clustering on the listing widened intervals 3–5× and brought coverage to **6/6**.

**3. My falsifiability test couldn't fail.** I wrote a test that patched the
post-treatment exclusion set and rebuilt the covariates, expecting an assertion
to fire. It could never fire — the same set filters the feature list *and* backs
the assertion, so a planted name is removed before the check sees it. It passed
while proving nothing. I now exercise the guard directly and document that the
real protection is an allow-list.

### The honest limitations (volunteer these — it reads as senior)

- **Ignorability holds in this dataset because the generator's targeting is
  observable.** That's a property of the data, not an achievement of the method.
  Real merchandiser judgement isn't in any table.
- **ROI on this dataset is a generator artefact.** Spend is ~20× the achievable
  margin at product-store grain, so 96.8% of events read as value-destroying and
  ROI bunches between −1.0 and −0.84. The arithmetic is tested; the numbers
  aren't interpretable. Step 8 needs spend at the right grain.
- **Cannibalisation isn't deducted**, so profit is an upper bound on category
  profit.
- **DiD passed its pre-trend test (p=0.64) and was still 21 points out.**

## Steps 8–9 — Elasticity and optimisation

**Elasticity** is log-log panel regression with product and store-time fixed
effects, which is the *correctly specified* model by construction — the
generator builds demand log-additively, so this is not a modelling guess. Four
estimators are implemented; two (`naive_ols`, `iv_2sls`) are computed but marked
not-selectable, so the agent can quantify the bias it avoided rather than assert
one exists.

Have this ready: **2SLS failed despite a median first-stage F of 484.** The
commodity cost index varies only at category×date, so it has no cross-sectional
variation to identify with. A strong F statistic is not sufficient for a valid
instrument, and the diagnostic that catches it checks for zero cross-sectional
variation directly.

**Optimisation** allocates a trade budget across candidate promotions by
incremental profit, subject to constraints, with diminishing returns modelled so
the budget spreads rather than pouring into one cell. Infeasibility is returned
as a *finding* — "your minimum spends already exceed the budget" — not an error.

Two bugs worth telling:

- The price grid was ±15%. The optimum for an elasticity of −2 is +20%, so
  **every recommendation pinned to the grid edge** and looked like a
  recommendation rather than a boundary artefact. Widened to ±30% with an
  explicit warning when the answer still lands on the edge.
- A constraint matching no candidate was silently dropped **and then reported as
  binding** — the worst combination, because it claimed to have constrained
  something it had never seen.

## Steps 10–12 — The agent layer

**Structured output via forced tool-calling**, not JSON parsing. The target
Pydantic model's JSON schema becomes a single tool's input schema and the model
is forced to call it. Materially more reliable than asking for JSON and hoping.

**A deterministic offline stub provider** implements the same ABC. Every agent
test, CI run and the golden-set evaluation go through it: no key, no network, no
cost, same answer every time. A suite that costs money per run gets run less
often, and a non-deterministic one produces failures nobody can reproduce.

**The Critic is a separate agent** because one that plans an investigation and
judges its own investigation will usually find that it succeeded. Mechanical
checks run first at no token cost; a result carrying `validation_status: failed`
is marked BLOCKING and overrides whatever the model concludes.

**Re-planning is bounded twice**, by `max_replans` and by the budget. A Critic
that is never satisfied is the realistic way an agent loops forever — so the cap
ends it, and the unresolved objection travels into the recommendation's risks
and caps its confidence at 0.5 rather than arriving looking settled.

**Output validation** is the answer to "how do you stop it inventing numbers",
and it is worth being specific about the two bugs found by running it:

- The numeral regex **missed "1.43M" entirely.** The bare number is 1.43, which
  falls under the structural floor that skips years and small counts — so the
  commonest way of writing a large figure was never checked at all. The
  hallucination control had a hole in exactly the case it exists for.
- Exempting suffixes and percentages then flagged **"95% confidence interval"**
  as an invented figure. Confidence levels are now scoped out by value *and*
  surrounding words together.

## Step 13 — Agent evaluation (lead with this one)

Twenty questions **derived from** the scenarios Step 2 injected, not written by
hand. That record is the only place a right answer exists independently of the
thing being graded.

Four dimensions, never averaged into one: tool selection (with a fan-out
penalty), evidence, direction, abstention. An agent that selects perfectly and
then writes an unsupported conclusion should not score the same as one that
picks badly and reports honestly.

**The floor is a keyword planner.** A stub run grades whatever the stub was
scripted to return — script the right answer and the score measures the person
who wrote the script. Scripting a deliberately weak policy instead makes the
number real.

| | keyword floor |
|---|---|
| answerable mean | 0.833 |
| abstention mean | **0.000** |
| tool selection | 1.000 |
| direction | 0.773 |

**Say the uncomfortable row out loud.** Keyword routing matches a language model
on tool selection for well-posed single-capability questions. The LLM earns its
place on abstention — where the floor scores zero, because knowing your own
coverage takes judgement — and on the `bad_promo` trap, where uplift is genuinely
positive and the right answer is still *don't repeat it*.

Two scorer bugs found by running it, both of which measured the wrong thing:

- Scoring direction on incremental **profit** graded Step 7's known spend
  artefact rather than the agent. Spend runs ~20× achievable margin at this
  grain, so profit is negative even for promotions injected as successful, and
  all three scored zero. The scenario record fixes the *volume* sign.
- Scoring the trap from the tool's ROI field graded Step 7 again. The decision
  *not to repeat* exists only in the recommendation, so that is where it is read
  from.

The report also separates **artefact gaps** — a required tool that ran and found
no trained series for that product — from planning errors. Different failures,
different fixes.

## Steps 14–15 — Interface and packaging

One `InvestigationService` behind the CLI, API and UI, so a question answered in
the terminal is answered identically over HTTP. The Streamlit UI talks HTTP
rather than importing the container: the shortcut would make it a second
consumer of the internals, and an endpoint could break without the demo noticing.

Three decisions worth defending:

- **An investigation that gathered no usable evidence is `failed`, not
  `completed`** — even though the graph ran to the end without raising.
  Reporting an empty result as complete is how "we found nothing" becomes
  "there is no effect".
- **A failed investigation returns 200 with a `failed` status**, not a 5xx. The
  request was handled correctly. That distinction lets a caller tell "the
  platform is down" from "the evidence did not support an answer".
- `POST /scenario` **reports the levers it could not model** rather than dropping
  them. A projection that silently ignored the inventory change the caller asked
  about would answer a different question than the one posed.

One bug worth telling: the service minted a `trace_id`, stored it, then let the
graph mint a different one — so the stored trace and the returned outcome could
not be joined. A silent failure, because both ids look perfectly valid.

## Engineering discipline

| | |
|---|---|
| Tests | 1,013 total, 175 for uplift alone |
| Gates | ruff, mypy (strict, 183 files), bandit — all clean |
| Config | YAML → Pydantic, hashed into MLflow params |
| Logging | structlog, JSON, trace IDs, never business data at INFO |
| Determinism | seeded RNG streams per generator |

---

# PART 4 — Fifty questions

## Section A — Project and architecture (1–8)

**1. Walk me through this project.**
Use the Part 2 pitch. Land on: LLM reasons, deterministic models compute, and
I built the computation layer first.

**2. Why build the tool layer before the agents?**
An agent calling an unreliable tool doesn't produce an unreliable answer — it
produces a *confident* unreliable answer, because the LLM adds fluent
justification on top. The failure gets harder to detect, not easier. I also
wanted the tool contract stable before anything depended on it, so agents plug
into an envelope rather than into models directly.

**3. What are the three seams and why do they matter?**
`DataRepository` (storage-agnostic business logic), the `ToolResult` envelope
(uniform contract with provenance and recoverability), and the DI container
(environment-keyed, lazy). They matter because they're almost impossible to
retrofit — every call site would have to change.

**4. Why separate LLM reasoning from numerical computation?**
Three reasons. Determinism: the same question must give the same number, and an
LLM won't. Auditability: a trade promotion decision needs a number traceable to a
model version and a dataset version. And capability: an LLM cannot fit a
propensity model or compute an influence function; asking it to approximate one
produces a plausible number with no error bar.

**5. How would an agent know a tool result is unreliable?**
It's in the envelope. `status` may be `partial`, `warnings` is non-empty,
`validation_status` may be `failed`, and `error.recoverable` tells it whether
re-planning could succeed. For uplift specifically, a failed causal validation is
promoted to the *first* warning so a supervisor reading a truncated list can't
miss it.

**6. Why synthetic data?**
Because it's the only way to validate a causal model. There's no real dataset
with a known counterfactual. The generator draws every relationship first,
records it to a directory the application cannot read, and then generates sales —
so I can compare an estimate against truth instead of against plausibility.

**7. Doesn't synthetic data make everything too easy?**
It makes some things easier and it's honest about which. The recovery results
establish the estimator is correct *given the assumptions*; they don't establish
the assumptions hold in production. Those are different claims and the docs keep
them apart. Real retail has structure the generator doesn't reproduce —
competitor reactions, assortment changes, private merchandiser information.

**8. What's the noise floor and why does it matter?**
Comparing the generator's `mean_demand` to the realised `latent_units` gives
35.0% WMAPE of irreducible negative-binomial noise. So a 40% model is at 1.15×
the floor, not "60% accurate". Without it you can't tell an excellent model from
a mediocre one, and you'll waste months chasing error that isn't there.

## Section B — Agentic design (9–18)

*These are all built. Answer them in the present tense and offer to show the
code — the repo will be opened.*

**9. How is the multi-agent system designed?**
Three agents in the graph, deliberately not one per model. **Supervisor** —
intent, entity extraction, planning, tool selection, re-planning. **Critic** —
validation, contradiction detection, sufficiency. **Recommendation** — synthesis
and final business output. (Root-Cause was designed as a fourth and folded into
the Supervisor's observe step; a separate agent that only interprets results
already in state was a node boundary without a job.) The six analytical models
are *tools*, not agents, because they're deterministic; wrapping each in an LLM
would add a non-deterministic layer in front of a deterministic computation and
buy nothing but latency and tokens.

**10. Why LangGraph rather than a chain or plain function calling?**
The workflow has cycles. Plan → Act → Observe → Re-plan is a loop with a
conditional exit, and the exit condition is a judgement the Critic makes. Chains
are DAGs. LangGraph gives explicit state, conditional edges, and checkpointing,
which is also what makes human-in-the-loop possible — you need to interrupt,
persist, and resume.

**11. What's in the graph state?**
The user's question and parsed intent, the plan, results gathered so far as
`ToolResult` objects, the Critic's verdicts, remaining budget, and the trace id.
Keeping results as full envelopes rather than extracted numbers is deliberate:
the Critic needs the assumptions and warnings to judge sufficiency, not just the
value.

**12. How do you stop an agent looping forever?**
A budget tracker — `app/guardrails/budget.py` — carried in state, decremented per
tool call and per LLM call, with hard caps on iterations, tokens and wall clock.
When it's exhausted the Supervisor must produce a report that says the
investigation was truncated rather than a confident conclusion.

**13. How does the Critic work?**
It's given the question, the plan, the collected evidence, and the assumptions
attached to each result, and returns a structured verdict: sufficient / needs
more evidence / contradictory. If evidence is thin it names *what* is missing so
the Supervisor can re-plan a specific step rather than retrying blindly.

**14. How does human-in-the-loop work?**
`interrupt_before` on the recommendation node, with a checkpointer so the graph
can persist and resume. Interrupt *before*, not after: the point is to review the
evidence, not to rubber-stamp a conclusion already drafted. It fires when the
projected impact crosses `AGENT__HUMAN_APPROVAL_THRESHOLD`, applied to the
*magnitude* — recommending you give up ₹1M is exactly as consequential as
recommending you chase it. Without a checkpointer the flag is still set but
nothing blocks on it, and I'd be explicit that this is the difference between
"flagged for approval" and "gated on it".

**15. How do you stop the LLM inventing numbers?**
Four layers, and only one of them is a prompt. The system prompt forbids stating
any number that didn't come from a tool result. Structured outputs via forced
tool-calling constrain the shape. **Post-hoc validation extracts every numeral
in the recommendation and matches it against the tool results in state** —
allowing the rounding a readable sentence applies, so 1,427,355 written as
"1.43M" passes. And the tools refuse rather than guess.

It reports rather than blocks, for three reasons: legitimate non-tool numbers
exist (a year, a horizon, "the top 3"); arithmetic over sourced values is real
and not separable from invention without the reasoning the check doesn't have;
and a labelled figure tells a reviewer where to look where a suppressed sentence
tells them nothing. An unsourced figure caps confidence at 0.6 and is named in
the output.

**16. How do you evaluate the agent?**
A golden set of twenty questions **derived from** the scenarios the generator
injected — the labelled events are "successful promo", "bad promo", "stockout",
"competitor price cut", so the correct answer exists independently of the thing
being graded. That's the payoff of building the data platform first.

Four dimensions scored separately: tool selection, evidence, direction,
abstention. Nine of the twenty are questions the registered tools *cannot*
answer, and those are scored on whether the agent declines — confidence is the
failure there.

The part worth volunteering: the baseline is a deliberately weak keyword planner,
and it **ties the language model on tool selection** for well-posed questions. A
stub run otherwise grades whatever you scripted it to return, which measures the
person who wrote the script.

**17. How do you version prompts?**
`prompts/` with a registry and versioned files — `supervisor/v1.md`,
`critic/v1.md`. The version travels into MLflow params and into the trace, so a
result can always be attributed to the prompt that produced it. I committed the
structure at the first agent commit rather than retrofitting it, because
retrofitting means touching every call site.

**18. What does observability look like?**
structlog with JSON output, a trace id propagated through contextvars so every
log line in one investigation correlates, and metrics on tool latency, token
counts, and budget consumption. Business data never gets logged at INFO — token
counts, model, stop reason and prompt version instead.

## Section C — Causal ML (19–30) — your strongest section

**19. What's promo uplift?**
Incremental sales caused by a promotion — observed minus what would have sold
without it. The second term is a counterfactual: the promotion ran, so the world
where it didn't is unobservable.

**20. Why isn't "promo sales minus non-promo sales" valid?**
Two errors compounding, both pointing up. Selection — promotions are scheduled
into rising demand, so the comparison books the season as lift. And pull-forward
— shoppers buy ahead and buy less afterwards, and the naive window ends before
the dip. Measured: naive +123.5% against a true +65.9%.

**21. ATE vs ATT vs CATE?**
ATE is the average over everything. ATT is the average over what was actually
treated. CATE is the average within a covariate profile. I target the **ATT**
because it's the business question — "what did the promotions we ran achieve",
not "what if we promoted everything" — and because it needs overlap only on the
treated support, which is a weaker assumption.

**22. What is a propensity score, and what's the trap?**
`e(X) = P(T=1|X)`. Rosenbaum-Rubin: if treatment is ignorable given X, it's
ignorable given `e(X)` alone, collapsing a 30-dimensional problem to one
dimension. **The trap**: a propensity model isn't trying to predict treatment
well. Perfect discrimination is a disaster — it means the groups are perfectly
separable and no comparison exists. The target is *balance*, not AUC.

**23. What's positivity/overlap?**
Every treated unit had a non-zero chance of not being treated. Unlike
ignorability this *is* testable. Where `e` approaches 1 the ATT weight `e/(1−e)`
diverges — a control row at 0.98 gets weight 49 and becomes the entire weighted
mean. I trim to [0.02, 0.98], report the trimmed share and effective sample size,
and refuse past a threshold.

**24. Explain double robustness.**
AIPW is consistent if *either* the outcome model or the propensity model is
correctly specified — not both. Two chances instead of one. It's not magic; if
both are wrong the estimate is wrong. **What I can show it bought**: on
confounded data the propensity model plainly failed to balance — worst
standardised difference 0.38 against a 0.10 threshold — and AIPW still recovered
+65.2% against a true +63.3%. The outcome model carried it. That measurement is
why balance failure *blocks* IPW in my pipeline and only *warns* for AIPW.

**25. What's the parallel trends assumption?**
Absent treatment, treated and control would have moved together. **The answer
that matters**: on my confounded panel the pre-trend test did *not* reject
(p=0.64) and DiD was still 21 points out. Failing the test disqualifies DiD;
passing it doesn't vindicate it. The test has power against linear pre-trends and
little else.

**26. How did you handle stockouts?**
This is the subtle one. Stockout is a *post-treatment* variable — promotions
raise demand, demand outruns replenishment, so treated rows censor more (measured
9.8% vs 3.1%). Dropping them conditions on a consequence of treatment and removes
the highest-demand promotion days, biasing downward. I drop them anyway because a
censored outcome records availability rather than demand — but I restate the
estimand to "the effect on sales among days where stock was available", report
censoring per arm, and run a bracketing sensitivity.

**27. How did you prevent temporal leakage?**
Anchoring — trailing covariates for a treated row are measured as of the *event
start*, not the row's own date. On day five of a promotion a trailing 7-day mean
anchored at the row contains four days of the effect being estimated. Plus
post-treatment exclusion, and cross-fitting that holds out whole listings.

**28. Why exclude discount from the covariates?**
Because it's a mediator, not a confounder — a consequence of treatment.
Conditioning on it holds the price cut fixed across arms and blocks the larger
causal path. Measured: +17.7% instead of +71.3%. A plausible number, wrong by
four times.

**29. What's a placebo test?**
Shift the treatment window into a period where no promotion ran. The true effect
is zero by construction, so anything found is attributable to the method. I
re-run the *full* pipeline rather than reusing fitted models, and I drop the real
treated rows entirely — leaving them in the control pool would produce a large
negative "effect" for a mechanical reason. Measured: +2.11% against a +62.4% real
estimate.

**30. What would make you reject an uplift estimate?**
Blocking: placebo finds a material effect, overlap fails after trimming, balance
fails for a propensity-only method, or a data-quality FAIL like overlapping
promotions. Warning-but-investigate: sensitivity spread above half the estimate,
weight calibration outside [0.7, 1.4], censoring gap above five points. And one
that isn't a diagnostic — unexplained disagreement with the other five methods.

## Section D — Forecasting and ML engineering (31–38)

**31. Why direct multi-step rather than recursive?**
Feature-distribution collapse. Recursive feeds conditional means forward, which
drives rolling standard deviations and volatility toward zero, so by day 30 the
model sees inputs it never saw in training. Direct with `h` as a feature avoids
that entirely.

**32. Why is `h` drawn randomly rather than from a fixed grid?**
A grid makes the model's splits on `h` piecewise-constant, which shows up as a
visible staircase in the daily forecast path.

**33. How do you get prediction intervals?**
Split conformal, calibrated per horizon bucket on a dedicated fold — distribution
free, with a finite-sample coverage guarantee. The property that matters isn't
the guarantee though; it's that achieved coverage is *measured on test data and
reported whatever it is*. A 90% interval that covers 71% is a finding.

**34. Why WMAPE and not MAPE or RMSE?**
MAPE explodes on near-zero actuals, which retail has constantly. RMSE is
dominated by a few large SKUs. WMAPE is volume-weighted, so a 50% error on a hero
SKU counts more than a 50% error on a slow mover — which is how the business
experiences it.

**35. What's Forecast Value Added?**
Accuracy improvement over a trivial benchmark, in WMAPE percentage points. Mine
is +12.6pp over horizon-seasonal-naive at every bucket. It's the number that
answers "was the model worth building" — a model that can't beat seasonal naive
shouldn't ship regardless of its absolute accuracy.

**36. Tell me about a hard bug.**
XGBoost came in at 82.9% WMAPE with +58% bias while LightGBM was at 47% on
identical data. Cause: `min_child_weight` sums *Hessians*, and under
`count:poisson` the Hessian is approximately μ, so the threshold scales with
target level — the same value that's reasonable for a slow mover is enormous for
a hero SKU. LightGBM's `min_child_samples` counts rows. Scaling to `50 × mean(y)`
gave 46.3% at −1.8% bias.

**37. How do you prevent leakage in the feature layer?**
Availability classes — every feature is declared OBSERVED, KNOWN_IN_ADVANCE or
STATIC, and a `PointInTimeView` masks realised columns beyond the as-of date. Then
a mutation test: corrupt all observed data after a cutoff and assert training
features are byte-identical. Plus a falsifiability test that plants the bug and
asserts the mutation test *fails* — a check that has never failed is
indistinguishable from one that does nothing.

**38. Did hyperparameter tuning help?**
No, and I reported that. Nineteen trials, best 42.34% against a default 42.22%.
At 1.25× the irreducible noise floor there are only about nine WMAPE points of
learnable signal in total, and hyperparameters compete for a fraction of that.
The defaults shipped.

## Section E — LLMOps, guardrails, governance (39–45)

**39. What's your guardrail strategy?**
Layered. Input validation via typed Pydantic schemas. Execution budget with hard
caps. Output validation — numbers cross-checked against tool results. Tool-level
refusal — models return structured errors rather than guessing. And permission
tags on tools (`read_analytics`, `run_model`, `optimise`) so a guardrail layer can
gate which an agent may call.

**40. How do you handle hallucination specifically?**
The strongest control isn't a prompt, it's architecture: the LLM has no path to
produce a number. Every figure comes from a tool result with provenance attached.
The prompt reinforces it, structured outputs constrain the shape, and a validator
catches the rest.

**41. How would you version models and roll back?**
MLflow with a registered model per capability, `ModelMetadata` carrying name,
version, dataset version, feature version and an `approved` flag. An unapproved
model must never be served to an agent. Rollback is repointing the registry
alias; the config fingerprint in run params tells you whether two runs are even
comparable.

**42. How do you control cost?**
Prompt caching on the system prompt and tool specs — they're large, stable, and
re-sent every turn. Minimum-sufficient tool selection enforced in the Supervisor
prompt. Budget caps per investigation. And model tiering: a small model for
intent classification, the large one for synthesis.

**43. What does regression testing look like for an agent?**
A golden set of questions with known correct answers, seeded from the generator's
injected scenarios. Score plan quality (did it select the right tools), evidence
quality (did it gather what was needed), and answer correctness against ground
truth. Run on every prompt or model change — a prompt edit is a deployment.

**44. How do you make an agent explainable?**
The investigation trace *is* the explanation: the plan, each tool call with its
parameters, each result with provenance and assumptions, the Critic's verdicts,
and the synthesis. A user can ask "why this recommendation" and get the actual
chain rather than a post-hoc narrative.

**45. Where does governance live in Databricks?**
Unity Catalog for table and model grants and lineage, MLflow run ids on
registered models, the treatment definition in run params so two analyses can be
told apart, and an approval flag that gates agent access. Scheduled jobs fail on
a data-quality FAIL or a failed causal validation rather than publishing.

## Section F — The JD gaps: Azure, Neo4j, MCP, LangSmith (46–50)

*Be straightforward. Bridge from what you know. Do not claim these.*

**46. This project uses Claude and Databricks. We're an Azure shop — how do you
transfer?**
The architecture is deliberately provider-agnostic at the seam. `app/llm/base.py`
is an abstract `LLMProvider`; `claude.py` is one implementation and an
`AzureOpenAIProvider` is another — same `complete_structured` contract, same
forced-tool-calling approach for structured output. Azure OpenAI's function
calling and Anthropic's tool use are close enough that the provider is the only
file that changes. The same is true of storage: `DataRepository` already has
DuckDB and Databricks implementations, and Azure SQL or Fabric would be a third.
I haven't done the Azure migration, but the seams exist precisely so that
provider choice isn't an architectural commitment.

**47. Have you used Azure AI Search / vector search?**
I've built the abstraction and not an Azure implementation. `app/memory/base.py`
defines a `VectorStore` interface with Chroma for local and a Databricks Vector
Search implementation planned. Azure AI Search would be a third implementation —
and it's the strongest of the three for hybrid retrieval, because BM25 plus
vector with reciprocal rank fusion and semantic reranking is built in rather than
assembled. If you're asking whether I've run it in production: not yet. If you're
asking whether I understand what hybrid retrieval solves — keyword precision on
product codes and SKU names where embeddings are weak, plus semantic recall — yes,
and it's exactly the retail-catalogue problem.

**48. What about Neo4j and knowledge graphs?**
I haven't built one. Where I'd use it in this domain is clear though: the
product-relationship graph. My generator already has substitute and complement
edges with cross-price elasticities on them, and that's a graph — "which products
cannibalise this one" is a traversal, and it's awkward in SQL and natural in
Cypher. For GraphRAG the value is multi-hop questions that vector search answers
badly: "which suppliers are affected if this category's promotions stop" needs
edges, not similarity.

**49. Have you used MCP?**
Not in production. My tool layer is an internal equivalent — a registry with
typed schemas, a uniform result envelope, and permission tags — and MCP is the
standardisation of exactly that pattern across process boundaries. The migration
would be mechanical: each `AnalyticalTool` already declares a name, description,
input schema and output schema, which is the MCP tool spec. What MCP adds that I
don't have is the transport and the ability for tools to live outside the
application — which matters when tools are owned by different teams.

**50. LangSmith and OpenTelemetry?**
I've built structured tracing with correlated trace ids and metrics, but with
structlog rather than either of those. The concepts transfer directly — spans,
correlation, latency and token attribution. LangSmith's advantage is that it
understands LLM semantics natively: prompt versions, token costs, and evaluation
runs against datasets, rather than me building that on top of generic tracing.
For a team, the dataset-and-evaluation side is the part I'd adopt fastest,
because that's what makes prompt changes safe to ship.

---

# PART 5 — Closing notes

## Questions to ask them

- How much of the agent layer is built versus planned? Where's the hardest part
  right now?
- What's the evaluation story? How do you know a prompt change didn't regress?
- Is the numerical work done by models or by the LLM? *(This is your favourite
  question — you have a strong view and it opens your best material.)*
- What does HITL look like today — approval gates, or humans in the loop
  continuously?
- Who owns prompts? Is a prompt change a deployment?

## If you only remember five things

1. **LLM reasons, deterministic models compute.** Never blur it.
2. **+34.9% of pure fiction** — the naive method on promotions that did nothing.
3. **0.7 percentage points** — ground-truth recovery across 4,417 events.
4. **The −424% story** — shows you debug rather than accept.
5. **"A keyword planner ties the LLM on tool selection"** — volunteer the result
   that undercuts your own system. It is the single most senior thing you can
   say, and it is true.

## The tone that wins a Principal interview

Volunteer your limitations before you're asked. Say "the ROI numbers on this
dataset aren't interpretable and here's why" before they find it. Say "ignorability
holds here because the generator's targeting is observable — that's a property of
the data, not an achievement." Say "the Docker image is written but I've never
built it — Docker isn't installed on my machine." Senior engineers are trusted
because they mark their own work honestly, and almost nobody does it in
interviews.

The trap to avoid now that everything is built: the old version of this document
had you *underclaim*, and that was the right call then. It isn't now. Describe
the system in the present tense, then be precise about the edges — Stage 2 is
designed rather than built, agentic RAG deliberately isn't there, and the
evaluation says the model earns its place on judgement rather than on routing.
