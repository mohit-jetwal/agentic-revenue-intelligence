# Corforge — AI Lead / Principal AI Lead

Interview preparation. Read Part 1 before anything else.

---

# PART 1 — Positioning: what to claim and what not to

## The situation

The repository is at **Step 7 of a 23-step roadmap**. That is real, substantial,
senior-level work — but it is not the finished agentic platform your CV bullet
describes.

| Built and working | Designed, scaffolded, not implemented |
|---|---|
| Synthetic data platform with hidden ground truth | LangGraph graph, edges, checkpointing |
| Point-in-time-correct feature layer | Supervisor / Root-Cause / Critic / Recommendation agents |
| Baseline Sales model | Claude integration (`app/llm/claude.py` is a documented skeleton) |
| Demand Forecasting (XGBoost, conformal intervals) | Agentic RAG (vector store is a stub) |
| **Promo Uplift (causal: AIPW, DR-learner)** | Agent evaluation harness |
| `ToolResult` envelope + `AnalyticalTool` base | Trade-promo optimisation, elasticity, price optimisation |
| Two registered agent tools | Databricks (design documents only) |
| Prompt registry with versioned files | HITL approval flow |
| Budget guardrail, trace IDs, structured logging | Streamlit UI |
| FastAPI skeleton | |

**The prompt files exist but say "Placeholder" in their second line.** If the
interviewer opens `prompts/supervisor/v1.md` — and a Principal candidate should
expect their repo to be opened — that is what they see.

## Why the honest version is the stronger interview

Three reasons, and they are practical rather than moral.

**It survives a drill-down.** "Show me the state schema" has a great answer if
you designed one and a terrible answer if you claimed one you don't have. A
Principal interview *is* a drill-down.

**The gap itself is the senior story.** "I built the tool layer and the contracts
first, deliberately, because an agent calling an unreliable tool is worse than no
agent" is an architectural argument. Most candidates have the opposite problem —
a LangGraph demo with no defensible numbers underneath.

**The causal work is genuinely rare.** Most people applying for AI Lead roles
have RAG chatbots. Very few can explain why conditioning on discount would have
reported +17.7% instead of +71.3%, or why their confidence intervals were three
times too narrow until they clustered the standard errors. That is your
differentiator and it is completely true.

## The sentence to use

> "It's a 23-step build. I'm at step 7 — the deterministic tool layer and the
> contracts the agent layer consumes are done and validated; the LangGraph
> orchestration is designed and scaffolded but not yet implemented. I sequenced
> it that way on purpose: an agent that calls a tool returning a wrong number
> just launders that error into a business recommendation with a confident tone
> on top."

Then, if pressed on the CV bullet:

> "The CV describes the platform's target architecture, which I've designed end
> to end. What's running today is steps 1 through 7. I'd rather walk you through
> what I've actually measured than what I intend to build."

That answer lands well. It reads as someone who ships and knows the difference.

## What you must not do

- Do not describe running a LangGraph investigation. There isn't one.
- Do not quote agent evaluation metrics. None exist.
- Do not claim Azure anything. The project uses Anthropic Claude and is
  Databricks-targeted. There is no Azure in it.
- Do not claim Neo4j, MCP, LangSmith or OpenTelemetry. None are present.

For each of those, Part 4 Section F gives you a real answer.

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
> **What I've built.** Seven of twenty-three steps. A synthetic CPG dataset whose
> generator records the true relationships in a directory the application
> physically cannot read, so every model can be scored against known truth. A
> point-in-time-correct feature layer. Then three models: baseline sales, demand
> forecasting, and promotional uplift.
>
> **The hardest one.** Promo uplift is causal, not predictive. You're estimating
> what *would* have happened without the promotion — a quantity that is absent
> from every dataset that will ever exist. So you can't validate it by holding
> out data. I built a generator where the effect is known exactly, and the
> estimator recovers it in six of six scenarios. On the real dataset, validated
> against the generator's own recorded parameters across 4,417 promotion events:
> expected +71.3%, estimated +72.0%. Error of 0.7 percentage points.
>
> **What's next.** Steps 13 to 20 are the agent layer — tool interfaces are done,
> the Claude provider and LangGraph graph are designed and scaffolded. The tool
> contract is already in place so agents plug into a stable envelope rather than
> the models directly.

Stop there. Let them pick the thread.

---

# PART 3 — End-to-end walkthrough

## The architecture in one picture

```
┌──────────────────────────────────────────────────────────┐
│  AGENT LAYER          Supervisor · Root-Cause · Critic    │  ← designed,
│                       · Recommendation (LangGraph)        │    not built
├──────────────────────────────────────────────────────────┤
│  TOOL CONTRACT        AnalyticalTool → ToolResult         │  ← BUILT
│                       status · provenance · confidence    │
│                       · assumptions · warnings · trace_id │
├──────────────────────────────────────────────────────────┤
│  SERVICE LAYER        BaselineService · ForecastingService│  ← BUILT
│                       · PromoUpliftService                │
├──────────────────────────────────────────────────────────┤
│  MODEL LAYER          baseline · forecasting · uplift     │  ← BUILT
│                       (+ 5 more designed)                 │
├──────────────────────────────────────────────────────────┤
│  FEATURE LAYER        FeatureEngineer, availability       │  ← BUILT
│                       classes, PointInTimeView            │
├──────────────────────────────────────────────────────────┤
│  DATA ACCESS          DataRepository (ABC)                │  ← BUILT
│                       DuckDB local │ Databricks prod      │
├──────────────────────────────────────────────────────────┤
│  STORAGE              Parquet gold tables                 │  ← BUILT
└──────────────────────────────────────────────────────────┘
```

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

## Engineering discipline

| | |
|---|---|
| Tests | 777 total, 175 for uplift alone |
| Gates | ruff, mypy (strict, 152 files), bandit — all clean |
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

*Answer these as design, stated as design. That's legitimate and expected for
work in progress.*

**9. How is the multi-agent system designed?**
Four agents, deliberately not one per model. **Supervisor** — intent, planning,
tool selection, re-planning. **Root-Cause** — hypothesis generation and evidence
interpretation. **Critic** — validation, contradiction detection, sufficiency.
**Recommendation** — synthesis and final business output. The eight analytical
models are *tools*, not agents, because they're deterministic; wrapping each in
an LLM would add a non-deterministic layer in front of a deterministic
computation and buy nothing but latency and tokens.

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

**14. How would you do human-in-the-loop?**
LangGraph interrupt-before on the nodes that produce an outward-facing
recommendation. The graph checkpoints, a human sees the plan or the draft
recommendation with its evidence and assumptions, and approves, edits or rejects.
Approval is recorded against the trace id so the decision is auditable.

**15. How do you stop the LLM inventing numbers?**
Four layers. The system prompt forbids stating any number that didn't come from a
tool result. Structured outputs via forced tool-calling constrain the shape.
Post-hoc validation checks every numeral in the output against the tool results
in state. And the tools themselves refuse rather than guess — the uplift service
returns a structured refusal with a `recoverable` flag instead of a number when
the causal assumptions fail.

**16. How would you evaluate the agent?**
Two levels. Component: does the Supervisor select the right tools for a known
question, does the Critic catch a planted contradiction. End-to-end: a golden set
of questions seeded from the generator's injected scenarios — the data has
labelled events like "successful promo", "bad promo", "competitor price cut", so
I know the correct root cause and can score whether the agent found it. That's
the payoff of building the data platform first.

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
5. **"I'm at step 7 of 23"** — say it early, say it plainly, then show what
   works.

## The tone that wins a Principal interview

Volunteer your limitations before you're asked. Say "the ROI numbers on this
dataset aren't interpretable and here's why" before they find it. Say "ignorability
holds here because the generator's targeting is observable — that's a property of
the data, not an achievement." Senior engineers are trusted because they mark
their own work honestly, and almost nobody does it in interviews.
