"""Generate the architecture and workflow diagrams for the project handbook.

Matplotlib rather than Graphviz or Mermaid, deliberately: it is already a project
dependency, so the diagrams regenerate on any machine that can run the test
suite, with no system package to install and no rendering service to reach.

Every diagram is drawn from the same small vocabulary of helpers below, so the
whole set stays visually consistent - which matters more for a reference document
than any individual picture does.

    uv run python docs/generate_handbook_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT = Path(__file__).parent / "images"

# One palette for every figure. Colour carries meaning consistently across the
# document: blue is local/Stage 1, green is production/Stage 2, amber is a seam
# or decision point, red is a hazard or something excluded.
LOCAL = "#3b6ea5"
PROD = "#2e7d5b"
SEAM = "#c9821b"
HAZARD = "#b3403a"
NEUTRAL = "#5a6472"
LIGHT = "#eef2f7"
PAPER = "#ffffff"


def _canvas(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    figure, axes = plt.subplots(figsize=(width, height))
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")
    figure.patch.set_facecolor(PAPER)
    return figure, axes


def box(
    axes: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    colour: str = LOCAL,
    fill: str = LIGHT,
    fontsize: int = 9,
    bold: bool = False,
    text_colour: str | None = None,
) -> tuple[float, float]:
    """Draw a rounded box, return its centre."""
    axes.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.5",
            linewidth=1.6, edgecolor=colour, facecolor=fill,
        )
    )
    axes.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color=text_colour or "#1b2430",
        fontweight="bold" if bold else "normal",
        linespacing=1.45, wrap=True,
    )
    return x + w / 2, y + h / 2


def arrow(
    axes: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = NEUTRAL,
    style: str = "-|>",
    dashed: bool = False,
    label: str = "",
    label_offset: tuple[float, float] = (0, 2),
    fontsize: int = 8,
    curve: float = 0.0,
) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle=style, mutation_scale=14,
            linewidth=1.4, color=colour,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=2, shrinkB=2,
        )
    )
    if label:
        axes.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label, ha="center", va="center", fontsize=fontsize,
            color=colour, style="italic",
        )


def title(axes: plt.Axes, text: str, subtitle: str = "") -> None:
    axes.text(50, 96, text, ha="center", va="top", fontsize=13, fontweight="bold",
              color="#1b2430")
    if subtitle:
        axes.text(50, 90.5, subtitle, ha="center", va="top", fontsize=9,
                  color=NEUTRAL, style="italic")


def save(figure: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(figure)
    print(f"  {path.name}")


# ---------------------------------------------------------------------------
# 1. Two-stage architecture
# ---------------------------------------------------------------------------

def diagram_two_stage() -> None:
    figure, axes = _canvas(13, 7.5)
    title(axes, "Two-stage architecture",
          "The same business logic runs in both columns. Only the three seams are swapped.")

    axes.add_patch(FancyBboxPatch((3, 8), 42, 76, boxstyle="round,pad=1",
                                  linewidth=1.2, edgecolor=LOCAL, facecolor="#f7fafd"))
    axes.add_patch(FancyBboxPatch((55, 8), 42, 76, boxstyle="round,pad=1",
                                  linewidth=1.2, edgecolor=PROD, facecolor="#f5fbf8"))
    axes.text(24, 85, "STAGE 1  -  Local MVP", ha="center", fontsize=11,
              fontweight="bold", color=LOCAL)
    axes.text(76, 85, "STAGE 2  -  Databricks", ha="center", fontsize=11,
              fontweight="bold", color=PROD)

    rows = [
        ("Agent layer\nClaude + LangGraph", "Agent layer\nClaude + LangGraph", False),
        ("Tool contract\nAnalyticalTool / ToolResult", "Tool contract\n(unchanged)", False),
        ("Services\nbaseline, forecasting", "Services\n(unchanged)", False),
        ("Models\nLightGBM / XGBoost", "Models\n(unchanged)", False),
        ("DataRepository\nDuckDB + Parquet", "DataRepository\nDatabricks SQL + Delta", True),
        ("MLflow\nsqlite:///mlflow.db", "MLflow\nDatabricks + Unity Catalog", True),
        ("Vector store\nChroma", "Vector store\nDatabricks Vector Search", True),
    ]

    y = 74
    for left, right, is_seam in rows:
        colour = SEAM if is_seam else LOCAL
        fill = "#fdf6e9" if is_seam else LIGHT
        box(axes, 6, y, 36, 7.5, left, colour=colour, fill=fill, fontsize=8)
        box(axes, 58, y, 36, 7.5, right, colour=SEAM if is_seam else PROD,
            fill="#fdf6e9" if is_seam else "#eaf4ee", fontsize=8)
        if is_seam:
            arrow(axes, (43.5, y + 3.75), (57, y + 3.75), colour=SEAM, dashed=True)
            axes.text(50, y + 6.2, "SEAM", ha="center", fontsize=7,
                      color=SEAM, fontweight="bold")
        y -= 9.4

    axes.text(50, 4, "An environment variable selects the column: APP__ENVIRONMENT=local | databricks",
              ha="center", fontsize=8.5, color=NEUTRAL, style="italic")
    save(figure, "01_two_stage_architecture")


# ---------------------------------------------------------------------------
# 2. Layered component architecture
# ---------------------------------------------------------------------------

def diagram_layers() -> None:
    figure, axes = _canvas(12, 8)
    title(axes, "Layered architecture",
          "Each layer talks only to the one below it. Dependencies point downward, always.")

    layers = [
        (78, "Agent layer  (Steps 13-20)", "Supervisor - Critic - Memory - LangGraph workflow", "#e8eef6", LOCAL),
        (66, "Tool contract  (Step 1)", "AnalyticalTool  |  ToolResult  |  ToolRegistry  |  permissions", "#fdf6e9", SEAM),
        (54, "Service layer  (Steps 4-5)", "BaselineSalesService   -   ForecastingService", "#eef2f7", NEUTRAL),
        (42, "Model layer  (Steps 4-11)", "baseline  -  forecasting  -  elasticity  -  uplift  -  optimisation", "#eef2f7", NEUTRAL),
        (30, "Feature layer  (Step 3)", "FeatureEngineer  |  FeatureRepository  |  contracts + catalogue", "#eef2f7", NEUTRAL),
        (18, "Data access  (Step 3)", "DataRepository  |  PointInTimeView  |  availability classes", "#fdf6e9", SEAM),
        (6, "Storage  (Step 2)", "Parquet gold tables  +  hidden ground truth", "#eef2f7", LOCAL),
    ]

    for y, name, detail, fill, colour in layers:
        axes.add_patch(FancyBboxPatch((8, y), 84, 9.5,
                                      boxstyle="round,pad=0.5,rounding_size=1.2",
                                      linewidth=1.5, edgecolor=colour, facecolor=fill))
        axes.text(11, y + 6.3, name, fontsize=9.5, fontweight="bold", color="#1b2430")
        axes.text(11, y + 2.8, detail, fontsize=8, color=NEUTRAL)

    for y in (76, 64, 52, 40, 28, 16):
        arrow(axes, (50, y), (50, y - 4.5), colour="#98a2b0")

    axes.text(95, 42, "depends\ndownward", ha="center", fontsize=7.5,
              color=NEUTRAL, style="italic", rotation=90)
    save(figure, "02_layered_architecture")


# ---------------------------------------------------------------------------
# 3. Data generation (Step 2)
# ---------------------------------------------------------------------------

def diagram_data_generation() -> None:
    figure, axes = _canvas(13, 7.5)
    title(axes, "Step 2  -  Synthetic data with hidden ground truth",
          "A structural causal model, so the true answers exist and can be checked against.")

    box(axes, 4, 70, 22, 12, "Hidden parameters\nelasticity, promo lift,\ncross-price, seasonality",
        colour=HAZARD, fill="#fbeceb", fontsize=8, bold=True)
    box(axes, 4, 50, 22, 14, "Six confounders\nprice endogeneity\ncost instrument\npromo targeting\nstockouts",
        colour=HAZARD, fill="#fbeceb", fontsize=7.5)

    box(axes, 33, 62, 22, 12, "Latent demand\nlog-additive model\nNegBinomial draw", colour=LOCAL, fontsize=8.5)
    box(axes, 33, 44, 22, 10, "Inventory\n(s, S) policy", colour=LOCAL, fontsize=8.5)
    box(axes, 62, 53, 24, 12, "observed = min(\n  latent, inventory)",
        colour=SEAM, fill="#fdf6e9", fontsize=9, bold=True)

    box(axes, 62, 30, 24, 11, "Gold tables\nsales, pricing,\npromotions, inventory", colour=LOCAL, fontsize=8)
    box(axes, 62, 12, 24, 13, "ground_truth/\nlatent_units\nmean_demand\nlost_units",
        colour=HAZARD, fill="#fbeceb", fontsize=8, bold=True)

    arrow(axes, (26, 76), (33, 70), colour=HAZARD)
    arrow(axes, (26, 57), (33, 66), colour=HAZARD)
    arrow(axes, (55, 68), (62, 61), colour=NEUTRAL)
    arrow(axes, (55, 49), (62, 56), colour=NEUTRAL)
    arrow(axes, (74, 53), (74, 41), colour=NEUTRAL)
    arrow(axes, (74, 30), (74, 25), colour=HAZARD)

    axes.add_patch(FancyBboxPatch((30, 8), 26, 16, boxstyle="round,pad=0.6,rounding_size=1.5",
                                  linewidth=1.6, edgecolor=HAZARD, facecolor="#fbeceb",
                                  linestyle="--"))
    axes.text(43, 19, "Unreachable by any\nrepository method", ha="center", va="center",
              fontsize=8, color=HAZARD, fontweight="bold")
    axes.text(43, 12.5, "Only the evaluation path\nreads it, from disk", ha="center",
              va="center", fontsize=7.5, color=NEUTRAL, style="italic")
    arrow(axes, (56, 17), (62, 17), colour=HAZARD, dashed=True)

    axes.text(50, 3, "The stockout row is what makes Steps 4-5 testable: latent >> observed, "
                     "and only a correct model recovers it.",
              ha="center", fontsize=8.5, color=NEUTRAL, style="italic")
    save(figure, "03_data_generation")


# ---------------------------------------------------------------------------
# 4. Point-in-time and availability classes
# ---------------------------------------------------------------------------

def diagram_point_in_time() -> None:
    figure, axes = _canvas(13, 7)
    title(axes, "Step 3  -  Availability classes",
          "'Clamp everything to as-of' is wrong. What is knowable depends on the table.")

    axes.plot([10, 90], [30, 30], color=NEUTRAL, linewidth=1.5)
    axes.plot([52, 52], [26, 74], color=HAZARD, linewidth=2, linestyle="--")
    axes.text(52, 76, "as-of date", ha="center", fontsize=9, color=HAZARD, fontweight="bold")
    axes.text(12, 26.5, "past", fontsize=8.5, color=NEUTRAL)
    axes.text(86, 26.5, "future", fontsize=8.5, color=NEUTRAL)

    rows = [
        (62, "OBSERVED", "sales, inventory, competitor prices", 10, 52, LOCAL,
         "clamped - the future does not exist yet"),
        (48, "KNOWN_IN_ADVANCE", "calendar, promotions, pricing", 10, 90, PROD,
         "readable forward - the plan is already made"),
        (36, "STATIC", "products, stores, customers", 10, 90, NEUTRAL,
         "no time dimension"),
    ]

    for y, name, tables, x0, x1, colour, note in rows:
        axes.add_patch(FancyBboxPatch((x0, y), x1 - x0, 8,
                                      boxstyle="round,pad=0.3,rounding_size=1",
                                      linewidth=1.5, edgecolor=colour,
                                      facecolor=colour, alpha=0.18))
        axes.text(x0 + 2, y + 5.2, name, fontsize=8.5, fontweight="bold", color=colour)
        axes.text(x0 + 2, y + 2, tables, fontsize=7.5, color=NEUTRAL)
        axes.text(92, y + 4, note, fontsize=7.5, color=NEUTRAL,
                  style="italic", ha="left", va="center")

    axes.text(50, 16, "This is what makes forecasting possible at all: the promotion calendar for "
                      "a future date\nis legitimately readable, while the sales for that date are not.",
              ha="center", fontsize=8.5, color="#1b2430")
    axes.text(50, 7, "clamp_window() in data/repositories/availability.py is the single place the rule is applied.",
              ha="center", fontsize=8, color=NEUTRAL, style="italic")
    save(figure, "04_availability_classes")


# ---------------------------------------------------------------------------
# 5. Feature engineering pipeline
# ---------------------------------------------------------------------------

def diagram_features() -> None:
    figure, axes = _canvas(13, 6.5)
    title(axes, "Step 3  -  Feature engineering order",
          "The sequence is load-bearing: breadth before depth, target-derived dropped last.")

    stages = [
        ("Load sales\nvia PointInTimeView", LOCAL),
        ("prepare_panel\nsort by keys + date", SEAM),
        ("Breadth\npromotions, inventory,\ncompetitor, product, store", LOCAL),
        ("Depth\nlags, rollings,\nprice, calendar", LOCAL),
        ("Trim warmup\n400 days", NEUTRAL),
        ("drop_target_derived\nrevenue, cost, profit", HAZARD),
    ]

    x = 3
    centres = []
    for text, colour in stages:
        centre = box(axes, x, 45, 14, 22, text, colour=colour,
                     fill="#fbeceb" if colour == HAZARD else LIGHT, fontsize=7.5)
        centres.append(centre)
        x += 16

    for i in range(len(centres) - 1):
        arrow(axes, (centres[i][0] + 7.4, 56), (centres[i + 1][0] - 7.4, 56))

    axes.text(50, 33, "Sorting must precede every shift. Target-derived columns must be dropped last,\n"
                      "because revenue = units x price recovers the target exactly.",
              ha="center", fontsize=8.5, color="#1b2430")

    box(axes, 20, 10, 26, 14, "shifted_group()\nrolling_on_shifted()\nperiods < 1 rejected",
        colour=SEAM, fill="#fdf6e9", fontsize=8)
    box(axes, 54, 10, 26, 14, "mask_censored()\nstockout units -> NaN\nso lags stay clean",
        colour=SEAM, fill="#fdf6e9", fontsize=8)
    axes.text(50, 3.5, "Every temporal feature routes through these helpers - "
                       "a lag can never see its own row.",
              ha="center", fontsize=8, color=NEUTRAL, style="italic")
    save(figure, "05_feature_pipeline")


# ---------------------------------------------------------------------------
# 6. Baseline vs forecast - the central distinction
# ---------------------------------------------------------------------------

def diagram_baseline_vs_forecast() -> None:
    figure, axes = _canvas(13, 6.8)
    title(axes, "Step 4 vs Step 5  -  the distinction that carries both",
          "Same machinery, opposite counterfactual. Conflating them is how uplift analysis goes wrong.")

    axes.add_patch(FancyBboxPatch((4, 20), 43, 60, boxstyle="round,pad=1",
                                  linewidth=1.4, edgecolor=LOCAL, facecolor="#f7fafd"))
    axes.add_patch(FancyBboxPatch((53, 20), 43, 60, boxstyle="round,pad=1",
                                  linewidth=1.4, edgecolor=PROD, facecolor="#f5fbf8"))

    axes.text(25, 75, "BASELINE  (Step 4)", ha="center", fontsize=11,
              fontweight="bold", color=LOCAL)
    axes.text(74, 75, "FORECAST  (Step 5)", ha="center", fontsize=11,
              fontweight="bold", color=PROD)

    axes.text(25, 68, "What WOULD have happened\nwithout the intervention?",
              ha="center", fontsize=9, style="italic", color="#1b2430")
    axes.text(74, 68, "What WILL happen\ngiven the plan?",
              ha="center", fontsize=9, style="italic", color="#1b2430")

    left = [
        "Features at date D",
        "Target at date D",
        "lag_1 is legitimate",
        "Excludes promotions",
        "Excludes stockouts",
        "Scored vs latent demand",
    ]
    right = [
        "Features at origin t",
        "Target at t + h",
        "lag_1 at t only",
        "INCLUDES planned promos",
        "Excludes stockout targets",
        "Scored vs held-out future",
    ]
    y = 60
    for baseline_row, forecast_row in zip(left, right, strict=True):
        axes.text(25, y, baseline_row, ha="center", fontsize=8.2, color=NEUTRAL)
        axes.text(74, y, forecast_row, ha="center", fontsize=8.2, color=NEUTRAL)
        y -= 6.2

    box(axes, 12, 6, 26, 10, "Step 4 baseline is a\nNOWCAST", colour=HAZARD,
        fill="#fbeceb", fontsize=9, bold=True)
    box(axes, 62, 6, 26, 10, "Step 5 needs the\nORIGIN / TARGET split", colour=PROD,
        fill="#eaf4ee", fontsize=9, bold=True)
    arrow(axes, (38, 11), (62, 11), colour=HAZARD,
          label="at T predicting T+30, you do not know T+29", label_offset=(0, 3.4))
    save(figure, "06_baseline_vs_forecast")


# ---------------------------------------------------------------------------
# 7. The horizon dataset - origin/target join
# ---------------------------------------------------------------------------

def diagram_horizon_dataset() -> None:
    figure, axes = _canvas(13, 7.2)
    title(axes, "Step 5  -  The horizon dataset",
          "One training row = (origin t, horizon step h, target = units at t+h)")

    axes.plot([8, 92], [76, 76], color=NEUTRAL, linewidth=1.4)
    for x, label in ((25, "origin  t"), (72, "target  t + h")):
        axes.plot([x, x], [73, 79], color="#1b2430", linewidth=2)
        axes.text(x, 82, label, ha="center", fontsize=9.5, fontweight="bold")
    arrow(axes, (27, 76), (70, 76), colour=SEAM,
          label="horizon step h  (1..90, drawn at random)", label_offset=(0, 2.6))

    box(axes, 6, 40, 32, 24,
        "ORIGIN SIDE\n\nlags, rollings, dynamics\ncompetitor price + gap\nprice position\npromotion history",
        colour=LOCAL, fontsize=8)
    axes.text(22, 34, "sales_daily is OBSERVED\n-> clamped to t", ha="center", va="top",
              fontsize=7.5, color=NEUTRAL, style="italic")

    box(axes, 56, 40, 32, 24,
        "TARGET SIDE   (h_ prefix)\n\nh_calendar, h_festival\nh_promotion_flag\nh_selling_price\nh_season",
        colour=PROD, fill="#eaf4ee", fontsize=8)
    axes.text(72, 34, "calendar / promotions / pricing\nare KNOWN_IN_ADVANCE", ha="center",
              va="top", fontsize=7.5, color=NEUTRAL, style="italic")

    arrow(axes, (22, 65), (24, 73), colour=LOCAL)
    arrow(axes, (72, 65), (72, 73), colour=PROD)

    box(axes, 33, 11, 34, 13, "y  =  units at t + h", colour=SEAM, fill="#fdf6e9",
        fontsize=10, bold=True)
    # Routed down the left edge of the target box: a straighter path would cross
    # the KNOWN_IN_ADVANCE caption sitting directly below it.
    arrow(axes, (58, 40), (64, 25), colour=SEAM, curve=-0.3)

    axes.text(50, 5, "The h_ prefix makes the split visible in the feature-importance table, "
                     "where a misplacement would otherwise hide.",
              ha="center", fontsize=8, color=NEUTRAL, style="italic")
    save(figure, "07_horizon_dataset")


# ---------------------------------------------------------------------------
# 8. Temporal split with embargo
# ---------------------------------------------------------------------------

def diagram_embargo() -> None:
    figure, axes = _canvas(13, 6)
    title(axes, "Step 5  -  Temporal split with an embargo",
          "Without the gap, a boundary training origin has its target inside the evaluation fold.")

    def band(y: float, label: str, segments: list[tuple[float, float, str, str]]) -> None:
        axes.text(6, y + 4, label, fontsize=9, fontweight="bold", va="center")
        for x0, x1, name, colour in segments:
            axes.add_patch(FancyBboxPatch((x0, y), x1 - x0, 8,
                                          boxstyle="round,pad=0.2,rounding_size=0.8",
                                          linewidth=1.3, edgecolor=colour,
                                          facecolor=colour, alpha=0.25))
            axes.text((x0 + x1) / 2, y + 4, name, ha="center", va="center",
                      fontsize=7.5, color="#1b2430")

    band(62, "WRONG", [
        (22, 48, "train", LOCAL), (48, 60, "calibration", SEAM),
        (60, 74, "validation", SEAM), (74, 94, "test", PROD),
    ])
    arrow(axes, (47, 60), (58, 60), colour=HAZARD, curve=-0.35)
    axes.text(52, 55, "a train origin's target\nlands inside calibration",
              ha="center", fontsize=7.5, color=HAZARD, fontweight="bold")

    band(28, "RIGHT", [
        (22, 42, "train", LOCAL), (42, 48, "embargo", HAZARD),
        (48, 56, "calibration", SEAM), (56, 62, "embargo", HAZARD),
        (62, 72, "validation", SEAM), (72, 78, "embargo", HAZARD),
        (78, 94, "test", PROD),
    ])
    axes.text(50, 20, "Embargo = max_horizon (90 days). It costs 90 days of origins at every "
                      "boundary, and that cost is the point.",
              ha="center", fontsize=8.5, color="#1b2430")
    axes.text(50, 12, "The check measures against the NEAREST evaluation fold. Measuring against "
                      "test alone looks safe even with no embargo,\nbecause calibration and "
                      "validation sit in between.",
              ha="center", fontsize=8, color=NEUTRAL, style="italic")
    save(figure, "08_embargo_split")


# ---------------------------------------------------------------------------
# 9. Training pipeline flow
# ---------------------------------------------------------------------------

def diagram_training_pipeline() -> None:
    figure, axes = _canvas(13, 7)
    title(axes, "Training pipeline",
          "The order is the correctness argument. Any reordering makes one number a self-report.")

    steps = [
        (4, 74, "1  Sample series\nstore-clustered,\npair-exact", LOCAL),
        (26, 74, "2  Build history\nFeatureEngineer\n+ mask_censored", LOCAL),
        (48, 74, "3  Horizon dataset\norigin / target\nself-join", SEAM),
        (70, 74, "4  Split\nwith embargo", SEAM),
        (4, 46, "5  Train candidates\nnaive, seasonal,\nLightGBM, XGBoost", LOCAL),
        (26, 46, "6  Calibrate\nper-bucket +\naggregate conformal", SEAM),
        (48, 46, "7  Score on test\nper horizon bucket", LOCAL),
        (70, 46, "8  Compare + select\naccuracy, then\nsimplicity", SEAM),
        (26, 18, "9  Persist model\n+ evaluation report", PROD),
        (48, 18, "10  Track to MLflow\n(failure is non-fatal)", PROD),
    ]
    centres = {}
    for x, y, text, colour in steps:
        centres[text[:4]] = box(axes, x, y, 20, 17, text, colour=colour,
                                fill="#fdf6e9" if colour == SEAM else LIGHT, fontsize=7.8)

    order = list(centres.values())
    for i in range(3):
        arrow(axes, (order[i][0] + 10.5, 82.5), (order[i + 1][0] - 10.5, 82.5))
    arrow(axes, (80, 73), (14, 64), colour="#98a2b0", curve=0.12)
    for i in range(4, 7):
        arrow(axes, (order[i][0] + 10.5, 54.5), (order[i + 1][0] - 10.5, 54.5))
    arrow(axes, (80, 45), (36, 36), colour="#98a2b0", curve=0.12)
    arrow(axes, (46.5, 26.5), (48, 26.5))

    axes.add_patch(FancyBboxPatch((70, 14), 26, 14, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  linewidth=1.5, edgecolor=HAZARD, facecolor="#fbeceb",
                                  linestyle="--"))
    axes.text(83, 21, "Report is written BEFORE\ntracking. A 3-hour Step 4 run\n"
                      "was lost to a tracking failure\nraised after training finished.",
              ha="center", va="center", fontsize=7.3, color=HAZARD)
    save(figure, "09_training_pipeline")


# ---------------------------------------------------------------------------
# 10. Serving path and the agent contract
# ---------------------------------------------------------------------------

def diagram_serving() -> None:
    figure, axes = _canvas(13, 7.2)
    title(axes, "Serving path and the agent contract",
          "The agent never learns that LightGBM exists, where the data lives, or what MLflow is.")

    box(axes, 4, 76, 22, 13, "Claude agent\n\"Forecast P001\nfor 30 days\"", colour=PROD,
        fill="#eaf4ee", fontsize=8.5, bold=True)
    box(axes, 34, 76, 26, 13, "ForecastingTool\ninput_schema\noutput_schema", colour=SEAM,
        fill="#fdf6e9", fontsize=8.5)
    box(axes, 68, 76, 26, 13, "run()  is @final\nvalidate, time, trace,\nwrap errors",
        colour=SEAM, fill="#fdf6e9", fontsize=8)

    box(axes, 34, 54, 26, 13, "ForecastingService\nvalidate as_of\nbuild response", colour=LOCAL, fontsize=8.5)
    box(axes, 68, 54, 26, 13, "FittedForecastModel\nscaffold + predict\n+ intervals", colour=LOCAL, fontsize=8.5)
    box(axes, 34, 32, 26, 13, "Future scaffold\ncross(pairs, dates)\n+ KNOWN_IN_ADVANCE", colour=LOCAL, fontsize=8)
    box(axes, 68, 32, 26, 13, "HorizonCalibration\nper-bucket +\naggregate", colour=SEAM,
        fill="#fdf6e9", fontsize=8.5)

    arrow(axes, (26, 82.5), (34, 82.5))
    arrow(axes, (60, 82.5), (68, 82.5))
    arrow(axes, (81, 76), (60, 67), colour=NEUTRAL, curve=0.15)
    arrow(axes, (60, 60.5), (68, 60.5))
    arrow(axes, (81, 54), (60, 45), colour=NEUTRAL, curve=0.15)
    arrow(axes, (60, 38.5), (68, 38.5))

    box(axes, 4, 8, 90, 16,
        "ToolResult:  status | model_name | model_version | dataset_version | result | "
        "confidence | assumptions | warnings | trace_id\n\n"
        "confidence is MEASURED interval coverage, or absent. Never a plausible-looking default.",
        colour=PROD, fill="#eaf4ee", fontsize=8.5)
    arrow(axes, (81, 32), (50, 25), colour=NEUTRAL, curve=0.15)
    save(figure, "10_serving_path")


# ---------------------------------------------------------------------------
# 11. Leakage defence in depth
# ---------------------------------------------------------------------------

def diagram_leakage() -> None:
    figure, axes = _canvas(12.5, 7)
    title(axes, "Leakage: defence in depth",
          "It never raises, never warns, and makes every number look better. Only assertions catch it.")

    rings = [
        (78, "Structural", "FeatureEngineer requires a PointInTimeView.\n"
                           "A bare repository raises TypeError.", LOCAL),
        (62, "Exclusion lists", "target-derived, supply, time_index, year, hive partition key", SEAM),
        (46, "Per-row arithmetic", "lag_7 reconstructed by hand from the source panel", SEAM),
        (30, "Behavioural", "error must grow with horizon; must not beat the noise floor", PROD),
        (14, "Falsifiability", "plant the exact bug, assert the detector FIRES", HAZARD),
    ]
    for y, name, detail, colour in rings:
        axes.add_patch(FancyBboxPatch((10, y), 80, 12,
                                      boxstyle="round,pad=0.4,rounding_size=1.2",
                                      linewidth=1.6, edgecolor=colour,
                                      facecolor=colour, alpha=0.14))
        axes.text(13, y + 8, name, fontsize=9.5, fontweight="bold", color=colour)
        axes.text(13, y + 3.5, detail, fontsize=8, color="#1b2430")

    axes.text(50, 7, "A test that has never failed proves nothing. The planted-bug test is what "
                     "makes the rest of the suite meaningful.",
              ha="center", fontsize=8.5, color=HAZARD, style="italic")
    save(figure, "11_leakage_defence")


# ---------------------------------------------------------------------------
# 12. Roadmap
# ---------------------------------------------------------------------------

def diagram_roadmap() -> None:
    figure, axes = _canvas(13, 5.5)
    title(axes, "Stage 1 roadmap",
          "Steps 1-5 complete. Each model becomes a tool; the agent layer arrives last.")

    done = [
        ("1", "Skeleton\n+ seams"), ("2", "Synthetic\ndata"), ("3", "Features\n+ contracts"),
        ("4", "Baseline\nsales"), ("5", "Demand\nforecast"),
    ]
    todo = [
        ("6", "Promo\nuplift"), ("7", "Price\nelasticity"), ("8", "Cross\nprice"),
        ("9-11", "Optimisation\n+ scenario"), ("12-15", "Tools +\nprompts"),
        ("16-20", "Agents +\nworkflow"),
    ]

    x = 3
    for number, label in done:
        box(axes, x, 45, 14, 24, f"Step {number}\n\n{label}", colour=PROD,
            fill="#eaf4ee", fontsize=8, bold=True)
        axes.text(x + 7, 40, "complete", ha="center", fontsize=7.5,
                  color=PROD, fontweight="bold")
        x += 16

    x = 3
    for number, label in todo:
        box(axes, x, 8, 14, 24, f"Step {number}\n\n{label}", colour=NEUTRAL,
            fill="#f2f4f7", fontsize=8)
        x += 16

    axes.text(50, 36, "Each completed step is consumed by the ones below it: "
                      "uplift measures against the baseline, optimisation starts from the forecast.",
              ha="center", fontsize=8, color=NEUTRAL, style="italic")
    save(figure, "12_roadmap")


def main() -> None:
    print("Generating handbook diagrams:")
    diagram_two_stage()
    diagram_layers()
    diagram_data_generation()
    diagram_point_in_time()
    diagram_features()
    diagram_baseline_vs_forecast()
    diagram_horizon_dataset()
    diagram_embargo()
    diagram_training_pipeline()
    diagram_serving()
    diagram_leakage()
    diagram_roadmap()
    print(f"\nWritten to {OUTPUT}")


if __name__ == "__main__":
    main()
