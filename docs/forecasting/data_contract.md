# Forecasting data contract

What the forecasting model consumes, what it guarantees about it, and what it
refuses to run on.

---

## 1. Grain and target

| | |
|---|---|
| **Grain** | `date × product_id × store_id` — one row per product, per store, per day |
| **Target** | `units` — units sold, an over-dispersed count |
| **Frequency** | daily |
| **Horizons served** | 7, 14, **28**, 30, 90 days |
| **Aggregation** | forecasts are produced at product × store and summed upward |

**28 days is the planning horizon**, and the reason is weekly seasonality. Four
whole weeks contain exactly four of each weekday, so the total is not skewed by
which days happen to fall inside the window. A 30-day window contains five of two
weekdays and four of the rest, which biases the total toward whichever two those
are — a real distortion when the weekly peak-to-trough swing is the largest
seasonal effect in the data.

## 2. Source tables and their availability class

The contract that makes forecasting possible at all. Not every table can be read
forward, and treating them uniformly would either leak or throw away the
information the forecast depends on.

| Table | Class | Readable for a future date? |
|---|---|---|
| `sales_daily` | `OBSERVED` | **No** — clamped to as-of |
| `inventory` | `OBSERVED` | **No** |
| `competitor_pricing` | `OBSERVED` | **No** |
| `calendar` | `KNOWN_IN_ADVANCE` | **Yes** |
| `promotions` | `KNOWN_IN_ADVANCE` | **Yes** — the plan is already made |
| `pricing` | `KNOWN_IN_ADVANCE` | **Yes** |
| `products`, `stores` | `STATIC` | Yes — no time dimension |

Enforced in one place: `clamp_window()` in `data/repositories/availability.py`.
A `PointInTimeView` applies it on every read, and `FeatureEngineer` raises
`TypeError` if handed a bare repository — so point-in-time correctness is a
construction that will not run rather than a discipline someone has to remember.

## 3. Known-future versus historical features

| Sourced at the **origin** `t` | Sourced at the **target date** `t+h` |
|---|---|
| lags (1, 7, 14, 28, 56, 364) | calendar: day of week, week, month, season |
| rolling means and standard deviations (7, 14, 28, 56) | holiday and festival flags, days to festival |
| demand momentum, volatility, trend | planned promotion flag, type, discount, duration |
| price position, discount depth, price index | planned selling and regular price |
| competitor price, gap, ratio | — |
| product and store attributes | — |

Target-side columns carry an **`h_` prefix**. That is not cosmetic: it makes the
split visible in the feature-importance table, which is the one place a
misplacement would otherwise hide behind a plausible-looking ranking.

The full per-feature table, with a derived leakage-risk rating for each, is in
[`feature_catalogue.md`](feature_catalogue.md).

## 4. Forbidden features

| Excluded | Why |
|---|---|
| `revenue`, `cost`, `gross_profit`, `sold_units` | `revenue = units × price` recovers the target exactly |
| all `inventory_*`, `stockout_*`, `received_units`, `sold_units_lag_1` | Step 4 **measured** this: with inventory features the model recovered only 0.30 of true stockout demand, having learned that low stock predicts low sales |
| `time_index` | anchored to the frame's own minimum, so the same calendar date differs between training and serving |
| `year`, `financial_year` | either a year the model has seen and overfits, or one it cannot place |
| `part` | hive partition key — a storage artifact that does not exist for a future date |
| `units_uncensored` | the masked target; a direct leak |

## 5. Row-level exclusions

| Rule | Reason |
|---|---|
| **Target on a stockout day → excluded** | The target there records availability, not demand |
| **Origin on a stockout day → kept** | A stockout at the origin is a legitimate knowable state; dropping those origins biases the feature distribution for no gain |
| **Origin without 364 days of history → excluded** | The seasonal lag is undefined; the model would train on mostly-NaN rows |
| **Origins inside the embargo band → excluded** | Their targets would land in the evaluation window |

Censored values are also **masked before lags are computed**
(`mask_censored`), so a stockout does not depress the next eight weeks of demand
history. Excluding the rows alone is not enough — the model would otherwise learn
the supply failure through the lag features instead.

## 6. Quality thresholds

Checked by `ml/forecasting/quality.py`, runnable as `uv run ari forecast-quality`.
Each check states its **forecasting** consequence, not a generic data-hygiene one.

| Check | Threshold | Severity | Why it matters here |
|---|---|---|---|
| duplicate `(product, store, date)` | 0 | **FAIL** | Doubles that day's weight in every lag and rolling window; the self-join then emits two training rows for one observation |
| missing dates | ≤ 2% | WARN → FAIL | A gap shifts every lag across it — `lag_7` reaches eight days back. Violates no schema, so nothing else would catch it |
| negative units | 0 | **FAIL** | Not a quantity, and undefined under a Poisson objective |
| non-positive price | 0 | **FAIL** | Breaks the price features and makes revenue derivation meaningless |
| missing target | 0 | WARN | Cannot train or score; makes every metric's denominator ambiguous |
| zero-sales share | ≤ 60% | WARN | Decides whether MAPE means anything |
| selling > regular price | 0 | WARN | Inverts the discount features — the model reads a promotion as a price rise |
| price jump > 10× overnight | 0 | WARN | A units or currency error, not a pricing decision |
| promotion flag without discount | 0 | WARN | Makes the flag a proxy for whatever else happened that day |
| discount without flag | 0 | WARN | An unlabelled promotion inflates the non-promotional baseline |
| promotion share | 0–60% | WARN | Too few and the effect is unlearnable; too many and the baseline has no support |
| stockout share | < 25% | WARN | Excluded rows are unusable history; a high share grows the excluded-tail bias |

**WARN means usable with a caveat. FAIL means forecasting from this panel would
produce numbers nobody should act on.** Collapsing the two levels would make the
report either alarmist or ignored.

Every check is tested twice — once against a clean panel where it must pass, and
once against a deliberately corrupted one where it must fire. A check that has
only ever returned PASS is indistinguishable from one that returns PASS
unconditionally.

## 7. Coverage limits

The generated dataset runs **2023-01-01 to 2025-12-31**, and the calendar,
promotion schedule and price plan all stop on the same day. A forecast therefore
needs its whole horizon to fall inside that window:

| Horizon | Latest usable as-of |
|---|---|
| 7 days | 2025-12-24 |
| 14 days | 2025-12-17 |
| 28 days | 2025-12-03 |
| 30 days | 2025-12-01 |
| 90 days | **2025-10-02** |

Past those dates the service **refuses** with a recoverable
`insufficient_data` error naming the latest as-of that would work. The
alternative — assume no promotion is planned and carry the last price forward —
produces a number that is systematically low and indistinguishable from a real
forecast.

## 8. Reproducibility

| Recorded | Where |
|---|---|
| `dataset_version` | from the generation manifest, on every response |
| `feature_version` | `features/contracts/specs.py` |
| `code_version` | `git rev-parse --short HEAD` |
| `config_fingerprint` | one hash over the entire `ForecastConfig` |
| seed, split boundaries, hyperparameters | MLflow params |

Two runs sharing a config fingerprint used the same setup. Two that differ are
not comparable — and the difference is discoverable rather than argued about.
