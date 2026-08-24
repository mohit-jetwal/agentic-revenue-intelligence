# Agentic Revenue, Pricing & Promotion Intelligence Platform

Agentic decision intelligence for CPG/Retail revenue, pricing and promotion
management. Claude plans, selects tools, interprets evidence and re-plans.
Deterministic ML, statistical and optimisation models produce every number.

> **Status: Stage 1, Step 1 — project skeleton.**
> The application starts, `/health` is live, and the architectural seams are in
> place. No data, models or agents yet. See [Roadmap](#roadmap).

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

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). No Docker, no cloud
account, no Databricks.

```powershell
git clone <repo> ; cd agentic-revenue-intelligence
.\tasks.ps1 setup          # uv sync --all-extras, creates .env
.\tasks.ps1 check          # ruff + mypy + pytest
.\tasks.ps1 api            # http://localhost:8000/docs
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
which step supplies it.

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
├── generation/     synthetic data generator (Step 2)
├── repositories/   Local (DuckDB/Parquet) + Databricks implementations
└── local/          generated output — git-ignored
databricks/         Stage 2. See databricks/README.md
prompts/            versioned prompts: <agent>/<version>.md
tests/              unit · integration · models · agents
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

## Deliberate scope decisions

Stated plainly, because "why isn't X here" is a fair question:

- **`/metrics` returns JSON counters, not Prometheus.** Nothing scrapes it in
  Stage 1, and Stage 2 monitoring is Databricks-native. Counters are
  process-local and reset on restart.
- **No DI framework.** Four factories and one switch. `dependency-injector`
  would add a dependency and a mental model in exchange for nothing.
- **No `aws/`, `terraform/`, `lambda/`, `ecs/`, `eks/`, `ecr/`.** Production is
  Databricks-native. There is no `boto3` in the dependency tree.
- **Empty tool registry.** Tools are registered in Step 13, once Steps 4–11
  supply the models. A tool becoming callable by an agent should be a decision
  someone made, not a side effect of a file existing.
- **Stage 2 classes are declared but raise.** Writing the signatures now proves
  the interfaces are satisfiable by a warehouse-backed store and surfaces any
  place the ABC leaked a local-only assumption.

---

## Roadmap

**Stage 1 — local MVP.** 1 skeleton ✅ · 2 synthetic data · 3 repository ·
4 baseline · 5 forecasting · 6 promo uplift · 7 trade-promo optimisation ·
8 price elasticity · 9 cross-price elasticity · 10 price optimisation ·
11 scenario engine · 12 MLflow · 13 tool interfaces · 14 Claude ·
15 LangGraph Supervisor · 16 agentic loop · 17 Critic · 18 re-planning ·
19 FastAPI · 20 Streamlit · 21 agent evaluation · 22 Docker ·
23 end-to-end validation

**Stage 2 — Databricks production.** Unity Catalog · Bronze/Silver/Gold ·
feature engineering · Databricks MLflow · Model Registry · Model Serving ·
Databricks SQL tools · Vector Search · production agents · security ·
monitoring · CI/CD

Step 2 note carried forward: the synthetic data must come from a **causal
simulation with known underlying parameters**, not independently sampled
columns. Price must actually move demand, stockouts must actually censor it, a
substitute's price must actually shift volume. Without that, every model in
Steps 4–11 is unfalsifiable; with it, each can be tested against the parameter
it is meant to recover. That test is the difference between a demo and evidence.

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
