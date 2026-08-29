# Agentic Revenue, Pricing & Promotion Intelligence Platform

Agentic decision intelligence for CPG/Retail revenue, pricing and promotion
management. Claude plans, selects tools, interprets evidence and re-plans.
Deterministic ML, statistical and optimisation models produce every number.

> **Status: Stage 1, Step 3 — data access, contracts and feature engineering.**
> The dataset is causally simulated with recoverable relationships, and models
> now reach it through a repository → contract → feature layer that makes
> future-data leakage **structurally impossible**. No models or agents yet.
> See [Roadmap](#roadmap).

---

## The idea

Seven analytical models with a chatbot bolted on is not an agentic system. The
point here is that the *investigation workflow itself* is dynamic: given a
question, the system decides what to analyse, in what order, whether the
evidence so far is sufficient, and whether to re-plan.

Two questions, two very different workflows:

```
"What is next month's forecast for Product A?"
    -> forecast tool -> validate -> answer.          One tool. Stop.

"Revenue fell 12%. Cut prices or promote harder?"
    -> baseline -> root cause -> elasticity -> cross-price
    -> promo uplift -> optimisation -> scenarios -> critic
    -> (re-plan if evidence is thin) -> recommendation
```

Fanning out to eight models for the first question would look impressive in a
demo and be wrong. Selecting the minimum sufficient workflow is the actual skill.

### The division of labour

| Claude does | Deterministic models do |
|---|---|
| Understand intent, set the objective | Forecast demand |
| Plan and re-plan the investigation | Estimate elasticity |
| Select tools | Measure promotional uplift |
| Interpret structured results | Optimise budget and price |
| Detect contradictions, judge sufficiency | Simulate scenarios |
| Synthesise the recommendation | Every single number |

**Claude never computes a business figure.** This is not enforced by asking it
nicely in a prompt — prompts are requests, not guarantees. It is enforced
structurally: an agent's only route to a number is `AnalyticalTool.run()`, which
is `@final` and always returns a `ToolResult` carrying the model name, model
version and dataset version that produced the figure. A tool cannot return a
bare float, and a number with no provenance cannot enter a recommendation.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). No Docker, no cloud account, no
Databricks.

The interpreter is pinned in [.python-version](.python-version) to **3.14**,
which is what the dependency set was resolved and verified against — including
LightGBM, XGBoost, MLflow, DuckDB and the OR-Tools GLOP solver. `requires-python`
stays at `>=3.12` so the package remains installable on 3.12; the pin only fixes
what `uv sync` builds, so everyone gets the same environment.

```powershell
git clone <repo> ; cd agentic-revenue-intelligence
.\tasks.ps1 setup                              # uv sync --all-extras, creates .env
.\tasks.ps1 check                              # ruff + mypy + pytest

uv run ari generate-data --profile dev --seed 42   # ~55s, 23.6M rows
uv run ari validate-data --profile dev             # invariants + relationship recovery

.\tasks.ps1 api                                # http://localhost:8000/docs
```

Without the task runner:

```powershell
uv sync --all-extras
Copy-Item .env.example .env
uv run pytest
uv run uvicorn app.main:app --reload --port 8000
```

Then:

```powershell
uv run ari config          # effective configuration, secrets redacted
uv run ari health          # per-dependency status
uv run streamlit run app/ui/streamlit_app.py
```

`/health` reports **degraded** on a fresh checkout. That is correct: no data has
been generated and no API key is set. It reports which dependency is missing and
which step supplies it — and flips `data_repository` to `ok` once you generate.

---

## The dataset

Generated from a **structural causal model with hidden ground-truth parameters**,
not by sampling columns independently. Elasticities, cross-price coefficients and
promotion response curves are drawn *first*, written to a `ground_truth/`
directory no model or agent can reach, and only then used to simulate sales.

That inversion is what makes the whole project falsifiable. Instead of "the model
produced −1.38, is that right?" with no way to answer, Step 8 will compare −1.38
against a known −1.42 and report the error.

| Profile | Products | Stores | `sales_daily` | Total rows | Time |
|---|---|---|---|---|---|
| `smoke` | 40 | 30 | 354K | 1.3M | ~4s |
| `dev` | 300 | 200 | 6.7M | 23.6M | ~55s |
| `stress` | 500 | 1,000 | ~44M | ~150M | on demand |

### What the data proves about itself

`uv run ari validate-data` runs 32 business invariants and 12 relationship tests.
At dev scale, seed 42:

| Test | Result |
|---|---|
| Elasticity recovered (panel FE, store + month) | **7.1%** median error vs truth |
| Naive OLS, no controls | **28.4%** median error — 4× worse |
| Cross-price signs (own price controlled) | **7/7** correct |
| Competitor effect | **+1.21** controlled for own price |
| Stockout suppression | **71.6%** inside injected windows |
| Censoring when in stock | **0.0%** |

The second row is the one that matters as much as the first. If an uncontrolled
regression recovered truth just as well, the data would be too easy and Step 8's
careful specification would be theatre.

Six confounders are deliberate: price endogeneity, cost pass-through (which
doubles as a **valid instrument**), randomised price tests (a clean
identification subset), promotion targeting, competitor–cost correlation, and
endogenous stockouts. Full detail in
[docs/data/simulation-design.md](docs/data/simulation-design.md); table and column
reference in [docs/data/data-dictionary.md](docs/data/data-dictionary.md).

### Layers

```
data/local/
├── gold/          clean, model-ready — what every model reads
├── bronze/        same tables + injected defects, for the DQ framework
├── ground_truth/  hidden parameters + true uncensored demand — unreachable
└── manifest.json  seed, config hash, row counts, injected-defect tallies
data/sample/       ~1,000-row CSV extracts, committed for browsing
```

Gold stays pristine so Steps 4–11 aren't fighting nulls in every model; bronze
gives the quality checks real defects to catch.

---

## The feature layer

```
DataRepository → PointInTimeView → FeatureEngineer → FeatureRepository → X, y
```

A model receives a `FeatureSet` and cannot tell whether it came from Parquet,
Delta or a Databricks Feature Table. That is the stated goal — but the property
that actually earns the layer is **point-in-time correctness**, because every
model in Steps 4–11 is temporal and a single leaked future value produces a model
that backtests beautifully and fails in production.

### Availability is per-table, not global

"Clamp everything to the as-of date" is the obvious approach and it is wrong. A
planner on 1 June genuinely knows the promotion calendar for 10–24 June, and
knows next Diwali's date. So each table is classified:

| Class | Tables | Visible past as-of? |
|---|---|---|
| `OBSERVED` | sales, inventory, competitor prices, trade promos | **No** |
| `KNOWN_IN_ADVANCE` | calendar, promotions, pricing | **Yes** — they're planned |
| `STATIC` | products, stores, customers, relationships | N/A |

The subtle half: a *future* promotion's schedule is knowable, but its realised
spend is not. Those columns are nulled beyond the as-of date.

### Leakage prevention is structural

```python
view = repo.as_of(date(2025, 12, 3))
engineer = FeatureEngineer(view)      # a bare repository raises TypeError
```

An `as_of_date` keyword can be forgotten — one omission in one builder and the
model trains on the future. A view has no method that returns future observed
data, so the mistake is unavailable rather than discouraged. Three more guards:

- **One shift helper.** `df.groupby(k).rolling(7)` reads fine and silently
  includes the current row. All temporal features route through
  `rolling_on_shifted`, which shifts first and rejects `periods < 1`.
- **Target-derived columns dropped.** `revenue = units × price`, so revenue plus
  price recovers the target exactly. Dropped centrally, not left to each model.
- **Forward-looking features declared.** Exactly three, each with a written
  justification, pinned by an allow-list the tests assert against.

### The test that matters

Build features twice — once from the full dataset, once from a dataset
*physically truncated* at the as-of date. For rows on or before that date, the
two must be **identical**. Any feature reaching forward makes them diverge.

It needs no knowledge of which feature leaked, so it keeps holding as Steps 4–11
add more.

```powershell
uv run pytest tests/leakage -v
```

### Contracts

Pandera schemas at the repository boundary (columns, dtypes, ranges, cross-column
identities), opt-in per call. Distinct from Step 2's `data/validation`, which
checks cross-row *business invariants* over a whole dataset — different problem,
different tool. Reversing Step 2's "no Pandera" decision is
[explained in the docs](docs/features/feature-catalogue.md).

Reference: [feature catalogue](docs/features/feature-catalogue.md) ·
[Databricks migration](docs/databricks_migration.md)

---

## Architecture

```
                          USER
                            |
                  FastAPI  /  Streamlit
                            |
                    SUPERVISOR AGENT
                  (plan / route / re-plan)
                            |
        +-------------------+-------------------+
        |                   |                   |
   Root Cause            Critic           Recommendation
     Agent               Agent                Agent
        |                   |                   |
        +-------------------+-------------------+
                            |
                     ANALYTICAL TOOLS
        (uniform ToolResult contract, versioned)
                            |
   baseline · forecast · uplift · elasticity · cross-price
        · trade-promo optimisation · price optimisation
                    · scenario simulation
                            |
                     DataRepository
                            |
        Stage 1: Parquet + DuckDB   |   Stage 2: Delta + Databricks SQL
```

Four agents, not twelve. The eight models are **tools**, not agents: they are
deterministic, and wrapping a deterministic computation in a non-deterministic
LLM layer buys latency and token cost in exchange for nothing.

### The three seams

Stage 1 runs locally; Stage 2 runs on Databricks. Section 44 of the brief
requires that migration not rewrite business logic. That is a property you
design in or you don't have, so it is designed in here:

| Seam | File | Guarantees |
|---|---|---|
| `DataRepository` | [data/repositories/base.py](data/repositories/base.py) | Models never know where data lives. Nothing outside this package imports `duckdb` or opens a file. |
| `ToolResult` | [app/schemas/tool_contract.py](app/schemas/tool_contract.py) | Every number reaches Claude with its model version, dataset version and confidence. |
| `Container` | [app/services/container.py](app/services/container.py) | One `APP__ENVIRONMENT` switch picks implementations. No `if databricks:` anywhere else. |

Flip `APP__ENVIRONMENT=databricks` today and you get an explicit
`ConfigurationError` naming the missing settings — not an `ImportError`, and
crucially not a silent fallback to local data, which would have an agent
answering from stale Parquet while believing it queried the warehouse.

### `agents/` vs `workflows/`

`app/agents/` holds node logic — prompts, parsing, the decision returned.
`app/workflows/` holds graph assembly — edges, routing, loops, checkpoints.
They change for different reasons: tuning how the Critic judges evidence should
not touch the graph, and adding a re-planning edge should not touch the Critic's
prompt.

---

## Layout

```
app/
├── api/            FastAPI app, middleware, routes
├── agents/         Supervisor, Root Cause, Critic, Recommendation
├── workflows/      LangGraph graph assembly
├── tools/          AnalyticalTool base + registry  <- contract enforcement
├── llm/            LLMProvider ABC, ClaudeProvider
├── memory/         VectorStore ABC (Chroma / Databricks Vector Search)
├── guardrails/     BudgetTracker; SQL & injection guards (Step 20)
├── observability/  structlog, trace context, metrics
├── schemas/        tool contract, agent state, API, domain enums
├── services/       Container (DI seam), ModelRegistry
├── config/         pydantic-settings
└── ui/             Streamlit demo
ml/                 8 model interfaces (baseline, forecasting, uplift,
                    elasticity, cross-price, both optimisers, scenario)
data/
├── generation/     causal simulation: generators, scenarios, ground truth
├── validation/     business invariants + relationship-recovery tests
├── contracts/      Pandera schemas, one per gold table
├── repositories/   Local (DuckDB/Parquet) + Databricks + point-in-time + sampling
├── sample/         committed CSV extracts
└── local/          generated output — git-ignored
features/
├── engineering/    lag · rolling · time · price · promo · inventory · entity
├── contracts/      feature specs, groups, model requirements, lineage
├── repositories/   Local (opt-in materialisation) + Databricks stub
└── datasets/       the five model-ready builders
configs/            data/ profiles · features/features.yaml
databricks/         Stage 2. See databricks/README.md
prompts/            versioned prompts: <agent>/<version>.md
notebooks/          data_validation/ · feature_validation/
docs/               data/ · features/ · databricks_migration.md
tests/              unit · integration · data · statistical · features · leakage
```

`app/`, `ml/` and `data/` are top-level packages per the project specification.
A `src/` layout would be the safer convention; the spec is explicit, so it is
followed — with generator *code* in `data/generation/` kept separate from
generated *files* in `data/local/`, which is the trap in that layout.

---

## Storage: two engines, one job each

| Job | Engine | Why |
|---|---|---|
| Analytical facts | Parquet + **DuckDB** | Every model runs full-column aggregations over the sales fact. DuckDB is columnar and reads only the columns touched; SQLite would scan whole rows. At the 1M–10M row scale the brief asks for, that is seconds versus minutes. |
| Application state | **SQLite** | Investigations, traces, feedback. Small, transactional, exactly SQLite's strength. |

Each maps to precisely one Databricks service in Stage 2 — see
[databricks/README.md](databricks/README.md).

---

## Configuration

Environment variables (or `.env`), nested as `SECTION__FIELD`. See
[.env.example](.env.example). Secrets are `SecretStr` and cannot leak through
`repr()`, a log line, or a settings dump — asserted in
[tests/unit/test_settings.py](tests/unit/test_settings.py).

Agent budgets (`AGENT__MAX_*`) bound the agentic loop. An agent that can re-plan
can loop forever, and a Critic that keeps returning "insufficient evidence" is
the realistic way it happens. On breach, the Supervisor returns the best
recommendation the gathered evidence supports, explicitly flagged as incomplete
— a truthful partial answer beats both an infinite loop and a confident answer
built on an investigation that never finished.

---

## How a wrong number is prevented from reaching a decision

The Supervisor plans and the tools compute, but neither is in a position to
judge whether the result answers the question. Three controls sit between the
evidence and the person reading it.

**The Critic is a separate agent.** An agent asked to plan an investigation *and*
assess whether its own investigation succeeded will usually find that it did.
Splitting the roles is the structural version of that scepticism. Its verdict
routes the graph: `valid=False` with a specific `required_followup` sends the
investigation back to planning with the gap named.

Before the model is consulted, `_mechanical_issues` runs checks that need no
judgement and cost no tokens. A result carrying `validation_status: failed` is
marked **BLOCKING** and overrides whatever the model concludes — that finding is
decidable, and no confident reading of it should be able to win.

**Re-planning is bounded twice**, by `AGENT__MAX_REPLANS` and by the budget check
inside `plan`. A Critic that is never satisfied is the realistic way an agent
loops forever, so the cap — not the Critic — ends the investigation. When it
does, the unresolved objection travels into the recommendation's risks and caps
its confidence at 0.5. An objection that was never answered must not arrive
looking settled.

**Every numeral in the final recommendation is checked against `tool_results`.**
This is the hallucination control that is architectural rather than prompted: the
system prompt tells the model never to state a number it did not get from a tool,
and [`app/guardrails/output_validation.py`](app/guardrails/output_validation.py)
verifies it did not. Figures are matched at the rounding a readable sentence
applies — 1,427,355 written as "1.43M", 0.6761 as "68%".

It reports rather than blocks, for three reasons. Legitimate non-tool numbers
exist (a year, a horizon, "the top 3"). Arithmetic over sourced values is real
and not distinguishable from invention without the reasoning the check does not
have. And a labelled number tells a reviewer where to look, where a suppressed
sentence tells them nothing. An unsourced figure caps confidence at 0.6 and is
named in the output.

**Impact above `AGENT__HUMAN_APPROVAL_THRESHOLD` requires a person.** With a
checkpointer the graph interrupts *before* the recommendation is written, not
after — the point is to review the evidence, not to rubber-stamp a conclusion
already drafted. Without a checkpointer the flag is still set but nothing blocks
on it, which is the difference between "flagged for approval" and "gated on it".

---

## Running it end to end

```powershell
uv run ari investigate "Did the promotion on product P00091 work?" --trace
uv run uvicorn app.main:app --reload          # then http://127.0.0.1:8000/docs
uv run streamlit run app/ui/streamlit_app.py  # in a second terminal
```

The CLI, the API and the UI all go through one `InvestigationService`, so a
question answered in the terminal is answered identically over HTTP. **The UI
talks HTTP rather than importing the container** — the shortcut would make it a
second consumer of the internals rather than a client of the API, and an
endpoint could then break without the demo noticing.

| endpoint | what it does |
|---|---|
| `POST /chat` | ask a question, get an answer and a recommendation |
| `POST /investigate` | same, with explicit product/store/date scope |
| `GET /investigation/{id}` | fetch a past investigation |
| `GET /investigation/{id}/trace` | how the answer was reached |
| `POST /scenario` | project levers directly, no agent round trip |
| `POST /feedback` | record whether a recommendation was useful |

Three decisions worth stating.

**An investigation that gathered no usable evidence is `failed`, not
`completed`** — even though the graph ran to the end without raising.
`completed` is a claim that the question was answered, and reporting an empty
result as complete is how "we found nothing" gets mistaken for "there is no
effect".

**A failed investigation returns 200 with a `failed` status**, not a 5xx. The
request was handled correctly; the investigation is what did not conclude. That
distinction is what lets a caller tell "the platform is down" from "the evidence
did not support an answer".

**`POST /scenario` reports what it could not model rather than dropping it.**
The API accepts an inventory lever and a promotion-spend amount that the
scenario engine has no way to project; those come back as warnings instead of
being silently ignored, because a projection that quietly dropped the change the
caller asked about would answer a different question than the one posed.

Investigations run synchronously. A background queue is the right answer at
production volumes and the wrong one here — it would add a worker, a result
store and a polling contract to serve a demo where the answer takes seconds.
`GET /investigation/{id}` already exists, so going async is a change of caller
behaviour rather than a rewrite.

---

## Grading the agent

```powershell
uv run ari evaluate-agent --provider stub --detail   # offline, free, deterministic
uv run ari evaluate-agent --provider claude          # costs money, hits the API
```

Every earlier step scored a *model* against ground truth. This scores the
*agent*: whether it picks the right tools, gathers what it needs, reads the
evidence the way the truth points, and declines when it cannot answer.

**The questions are derived, not invented.** Step 2 recorded exactly which
products, stores and windows each injected scenario touched, and
[`evaluation/golden_set.py`](evaluation/golden_set.py) builds 20 questions from
that record. A question written separately would be graded against an
expectation written separately, which measures nothing.

**Nine of the twenty are unanswerable, deliberately.** No registered tool
diagnoses a stockout, a competitor price cut or a lost-distribution shock. For
those the correct behaviour is to decline, and a confident answer is the
failure — so they are scored on abstention instead of accuracy, and the coverage
gap is reported rather than hidden by dropping the questions.

**Four dimensions, never averaged into one.** Tool selection (with a penalty for
fanning out), evidence, direction, abstention. An agent that selects perfectly
and then writes an unsupported conclusion should not score the same as one that
picks badly and reports honestly.

**The floor is a keyword planner.** A stub run grades whatever the stub was
scripted to return, so scripting the right answer would measure the person who
wrote the script. Instead the stub follows a policy — regex over the question
text, no reasoning — from
[`evaluation/baseline_planner.py`](evaluation/baseline_planner.py). That makes
the number a real measurement of a real, bad planner, and the bar a language
model has to clear to justify its cost.

Recorded in [`evaluation/baselines/`](evaluation/baselines/); `evaluate-agent`
exits non-zero when a headline number falls more than 0.02 below it.

| | keyword floor |
|---|---|
| answerable mean | 0.833 |
| abstention mean | **0.000** |
| tool selection | 1.000 |
| direction | 0.773 |

Two findings worth stating plainly. Keyword routing **matches a language model
on tool selection** for well-posed single-capability questions — the LLM does not
earn its place there. Where it does is the other two rows: the floor scores zero
on abstention because knowing your own coverage requires judgement, and it takes
half marks on the `bad_promo` trap, where uplift is genuinely positive and the
right answer is still *don't repeat it*.

The report also separates **artefact gaps** — a required tool that ran and found
no trained series for that product — from planning errors. They have different
fixes, and a data gap read as a reasoning gap sends the effort somewhere it
cannot help.

---

## Deliberate scope decisions

Stated plainly, because "why isn't X here" is a fair question:

- **`/metrics` returns JSON counters, not Prometheus.** Nothing scrapes it in
  Stage 1, and Stage 2 monitoring is Databricks-native. Counters are
  process-local and reset on restart.
- **No DI framework.** Four factories and one switch. `dependency-injector`
  would add a dependency and a mental model in exchange for nothing.
- **No `aws/`, `terraform/`, `lambda/`, `ecs/`, `eks/`, `ecr/`.** Production is
  Databricks-native. There is no `boto3` in the dependency tree.
- **Tools are registered explicitly, never discovered.** A tool becoming callable
  by an agent is a decision someone made, not a side effect of a file existing.
- **No Pandera or Great Expectations.** Most checks here are business invariants
  (`opening + received − sold = closing`) that schema libraries express awkwardly,
  and both are a large dependency for ~150 lines of arithmetic. A genuine
  trade-off — Pandera is the more conventional answer, and the right next step if
  the check set outgrows simple predicates.
- **Dataset config is YAML validated by Pydantic**, not pydantic-settings. ~100
  nested business parameters cannot be expressed as environment variables, and a
  change to elasticity bands must be visible in a diff. Environment config stays
  in `.env`; the two have different lifecycles and are deliberately not merged.
- **Stage 2 classes are declared but raise.** Writing the signatures now proves
  the interfaces are satisfiable by a warehouse-backed store and surfaces any
  place the ABC leaked a local-only assumption.

---

## Roadmap

**Stage 1 — local MVP.** Compressed from 23 steps to 15: the remaining
analytical models are merged, and the agent layer — the part the platform is
named for — is brought forward.

1 skeleton ✅ · 2 synthetic data ✅ · 3 data access & features ✅ ·
4 baseline ✅ · 5–6 demand forecasting ✅ · 7 promo uplift ✅ ·
8 price elasticity, own + cross ✅ · 9 optimisation & scenario engine ✅ ·
10 MLflow, Claude & stub providers ✅ · 11 LangGraph plan/act/observe loop ✅ ·
12 Critic, re-planning, human-in-the-loop ✅ · 13 agent evaluation ✅ ·
14 FastAPI & Streamlit ✅ · 15 Docker & end-to-end validation

Step 4 onward consumes `features/datasets/` — each model gets a builder that
already encodes its framing (which rows are eligible, what must be excluded to
keep the estimate honest) and is scored against Step 2's hidden ground truth.

The SQLite application-state store (investigations, traces, feedback) was
deferred from Step 3 and built in Step 14, once the API had investigations to
record. `DATA_BACKEND=postgres`
is documented as a switch point at `Container.data_repository` rather than
implemented, since a second analytical backend has no consumer.

**Stage 2 — Databricks production.** Unity Catalog · Bronze/Silver/Gold ·
feature engineering · Databricks MLflow · Model Registry · Model Serving ·
Databricks SQL tools · Vector Search · production agents · security ·
monitoring · CI/CD

Every model in Steps 4–11 will be scored against `ground_truth/`, using the same
method the Step 2 validation suite already applies to itself.

---

## Development

```powershell
.\tasks.ps1 lint       # ruff check
.\tasks.ps1 typecheck  # mypy
.\tasks.ps1 test       # pytest
.\tasks.ps1 security   # bandit
.\tasks.ps1 check      # all of the above
```

Docker arrives at Step 22. It is not installed on the development machine and is
not needed before then.
