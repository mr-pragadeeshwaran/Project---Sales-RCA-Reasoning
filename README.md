# 🧠 Advanced Sales RCA Engine — Documentation

> **Version:** 2.0 (Granular City × SKU × Day Causal Attribution)
> **Brand:** 24 Mantra Organic | **Platform:** Blinkit
> **Run:** `python advanced_rca_engine.py`

---

## 📋 Table of Contents

1. [What Does This Engine Do?](#what-does-this-engine-do)
2. [Data Sources](#data-sources)
3. [Configuration](#configuration)
4. [Architecture — 12 Layer System](#architecture--12-layer-system)
5. [Execution Flow](#execution-flow)
6. [Code Map — Where is Each Logic?](#code-map--where-is-each-logic)
7. [SKU × City Deep Dive (New in v2)](#sku--city-deep-dive-new-in-v2)
8. [How to Read the Output](#how-to-read-the-output)
9. [Output Files](#output-files)
10. [Key Metrics Explained](#key-metrics-explained)
11. [Thresholds Reference](#thresholds-reference)
12. [Common Questions](#common-questions)

---

## What Does This Engine Do?

Instead of just saying *"revenue dropped"*, this engine tells you **exactly WHY** it dropped — down to which specific city, which SKU, which driver (OSA / Discount / Competitor / Stock), and how much revenue (in ₹) each driver caused.

### Simple Analogy

Think of it like a doctor running tests:
- **L0** = Blood report (raw numbers, no interpretation)
- **L1–L5** = Diagnosis tests (trend, interactions, lags, thresholds)
- **L6–L7** = Was it the environment (market) or the patient (own-issue)?
- **L8** = Early warning scan (what will get worse?)
- **L9** = Final financial bill (₹ impact per driver)
- **L10** = Doctor's confidence level + recommended actions
- **L11–L12** = City × SKU surgery (granular causal drill-down)

---

## Data Sources

| File | Variable | What It Contains |
|------|----------|-----------------|
| `blinkit-availability-data-april26-day wise.csv` | `FILE1` | Store-level: OSA %, Stock, Listing %, Locality Sales Contribution per store per day |
| `blinkit-rca-download-April-26-Daily-City-Comp.csv` | `FILE2` | City-level: Revenue, OSA, Discount, Ad SOV, Category Share for ALL brands |

### How They Connect

```
FILE1 (Store Level)                FILE2 (City/Brand Level)
─────────────────────              ──────────────────────────
Store ID, Locality City  ───┐      Product ID, City, Brand
Product ID                  │      Revenue, OSA, Discount
Avg. OSA %, Stock Levels    │      Ad SOV, Category Share
Locality Sales Contribution ┘
         │
         ▼ Aggregated by (Date, City, Product ID)
         │
    Darkstore_Count        ──────► Merged into df24 (24M only)
    Network_Strength               Used for store quality signals
    High_Value_Darkstores          in SKU deep dive
```

---

## Configuration

Edit these constants at the **top of the file** (lines 17–43):

```python
# ── CONFIG ────────────────────────────────────────────────────────────
FILE1  = r"path\to\availability.csv"          # Store-level data
FILE2  = r"path\to\city-comp.csv"             # City + competitor data
OUTDIR = r"path\to\output"                    # Where reports are saved

COMPARE_DATE1 = "2026-04-19"   # Baseline day (the "good" day)
COMPARE_DATE2 = "2026-04-20"   # Analysis day (the "drop" day)

COMPARE_PERIOD1_START = "2026-04-14"  # Recent week start
COMPARE_PERIOD1_END   = "2026-04-20"  # Recent week end
COMPARE_PERIOD2_START = "2026-04-07"  # Prior week start
COMPARE_PERIOD2_END   = "2026-04-13"  # Prior week end
```

**To change comparison dates:** Just update `COMPARE_DATE1` and `COMPARE_DATE2`. Everything else recalculates automatically.

---

## Architecture — 12 Layer System

```
┌─────────────────────────────────────────────────────────────────┐
│                    advanced_rca_engine.py                        │
│                                                                   │
│  DATA IN ──► load_data() ──► build_ts() ──► build_locality_      │
│                                              intelligence()       │
│                                                  │                │
│              ┌───────────────────────────────────┘                │
│              ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  NATIONAL ANALYSIS (L0 → L10)                           │     │
│  │                                                         │     │
│  │  L0  Raw Snapshot      → Evidence base (last 7 days)   │     │
│  │  L1  Baseline          → Trend + Day-of-Week effects   │     │
│  │  L2  Interactions      → OSA×SOV, Discount×Network     │     │
│  │  L3  Thresholds        → SAFE / WARNING / CRITICAL     │     │
│  │  L4  Lag Analysis      → How many days before impact?  │     │
│  │  L5  Pullforward       → Fake spikes from promotions   │     │
│  │  L6  Competitive       → Market issue vs own issue?    │     │
│  │  L7  Geographic        → Which cities are bleeding?    │     │
│  │  L8  Leading Alerts    → What will get worse?          │     │
│  │  L9  Attribution       → ₹ impact per driver           │     │
│  │  L10 Confidence        → Trust score + actions         │     │
│  └─────────────────────────────────────────────────────────┘     │
│              │                                                    │
│              ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  GRANULAR ANALYSIS (L11–L12) — NEW in v2.0             │     │
│  │                                                         │     │
│  │  L11 SKU × City TS    → Per-SKU per-City time-series   │     │
│  │  L12 Causal Attribution → 30+ signals, dynamic OLS     │     │
│  │       ┌─── Top 5 losing SKUs                          │     │
│  │       └─── Per SKU: Top 4 dropping cities             │     │
│  │             └─── Per City: Markdown Table Output      │     │
│  └─────────────────────────────────────────────────────────┘     │
│              │                                                    │
│              ▼                                                    │
│  OUTPUT ──► JSON Report + Executive Summary MD + Excel           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Execution Flow

```
python advanced_rca_engine.py
         │
         ▼
1. make_output_dir()
   └─ Creates: output/RCA_April 23-2026 01-00 PM/

2. load_data()
   ├─ Reads FILE1 → df1 (1.9M rows, store level, 24M only)
   ├─ Reads FILE2 → df_all (467K rows, all brands)
   ├─ Filters 24M rows → df24 (11K rows)
   └─ Builds active_stores, dark store counts, Network_Strength

3. build_ts()
   └─ Aggregates df24 by Date → 20 rows national daily time-series

4. build_locality_intelligence(active_stores)
   └─ Groups stores: HHI, P75, High-Value count per city per SKU

5. build_sku_city_ts(df24, loc_stats)
   └─ Creates 11,415 rows: every SKU × City × Day combination

6. enrich_sku_city_ts()
   └─ Adds 15 columns: DoD changes, zone status, consecutive signals

7. L0 → L10 (National Layers)
   └─ Runs all diagnostic layers on national time-series

8. compare_two_days()
   ├─ Compares COMPARE_DATE1 vs COMPARE_DATE2
   ├─ Finds top 5 losing SKUs
   └─ Calls sku_deep_dive() for each loser ←── (NEW v2.0)

9. build_sku_city_attribution() × 2
   ├─ Day-level attribution (d1 vs d2)
   └─ Period-level attribution (week vs week)

10. export_llm_evidence_pack()
    └─ Saves Excel (9 sheets) + JSON evidence pack

11. generate_brief()
    └─ Writes advanced_rca_executive_summary.md
```

---

## Code Map — Where is Each Logic?

| What You Want to Find | Function Name | Line (approx) |
|----------------------|---------------|----------------|
| File paths & dates config | Constants block | 17–43 |
| Load CSV data | `load_data()` | 53–96 |
| Build national time-series | `build_ts()` | 99–116 |
| Raw daily snapshot (L0) | `L0_raw_snapshot()` | 120–157 |
| Trend + anomaly detection (L1) | `L1_baseline()` | 161–195 |
| Driver interaction checks (L2) | `L2_interactions()` | 242–264 |
| OSA/Discount/SOV zones (L3) | `L3_thresholds()` | 267–329 |
| Lag correlation engine (L4) | `L4_lag()` | 198–238 |
| Pullforward filter (L5) | `L5_pullforward()` | 332–347 |
| Market vs own issue (L6) | `L6_competitive()` | 350–450 |
| City breakdown (L7) | `L7_geographic()` | ~450–550 |
| Leading indicator alerts (L8) | `L8_leading()` | ~550–650 |
| Financial attribution (L9) | `L9_attribution()` | ~650–750 |
| Confidence scoring (L10) | `L10_feedback()` | ~750–850 |
| Day vs day comparison engine | `compare_two_days()` | ~886 |
| Store locality stats | `build_locality_intelligence()` | 1385–1416 |
| SKU × City time-series | `build_sku_city_ts()` | 1418–1462 |
| Enrich with 15 driver columns | `enrich_sku_city_ts()` | 1560–1629 |
| **Dynamic signal regression** | `_fit_sku_city_regression_dynamic()` | ~1634 |
| **SKU × City deep dive tables** | `sku_deep_dive()` | ~1710 |
| Historical proof lookup | `_get_historical_proof_city()` | 1710–1740 |
| Per-driver attribution row | `_build_attribution_row()` | 1743–1803 |
| Competitor snapshot | `_get_competitor_snapshot()` | 1833–1873 |
| Danger combination flags | `_detect_combinations()` | 1876–1905 |
| Full SKU × City attribution | `build_sku_city_attribution()` | 1908–2040 |
| Executive summary writer | `generate_brief()` | ~1192 |
| Main orchestrator | `main()` | 1325–1382 |

---

## SKU × City Deep Dive (New in v2)

This is the most important new feature. For each top-losing SKU, the engine drills into every dropping city and runs a localized causal test.

### How It Works — Step by Step

```
sku_deep_dive() called for e.g. "Moong Dal (Dhuli) 500g"
         │
         ▼
Step 1: Find dropping cities
   c_rev1 = revenue on d1 per city
   c_rev2 = revenue on d2 per city
   Dropping cities = cities where drop > ₹500
   Example: [Mumbai: -7840, Others: -6100, Hyderabad: -3200, Delhi: -1920]
         │
         ▼
Step 2: For each dropping city, build local daily time-series
   Merge signals from 3 sources:
   ┌─────────────────────────────────────────────────────┐
   │ SOURCE A: df24 (City/SKU level)                    │
   │   → Wt_OSA, Discount, Ad_SOV, Organic_SOV          │
   │                                                     │
   │ SOURCE B: df_all competitors (same category, city)  │
   │   → Comp_Discount, Comp_OSA, Comp_Ad_SOV           │
   │   → Comp_Disc_Adv = Comp_Discount - Own_Discount   │
   │   → Comp_Squeeze = Comp_Disc_Adv × (100-OSA)/100   │
   │                                                     │
   │ SOURCE C: df1 store-level (city filtered)           │
   │   → Stores, Store_OSA, Stock, Listing              │
   │   → HV_OSA, LV_OSA, HV_Stock, LV_Stock            │
   │   → Stock_roll3 (rolling 3-day avg stock)          │
   └─────────────────────────────────────────────────────┘
         │
         ▼
Step 3: Add structural signals
   Is_Monday = 1 if Monday, else 0
   DOW = day of week number
         │
         ▼
Step 4: Run _fit_sku_city_regression_dynamic()
   ┌─────────────────────────────────────────────┐
   │ For each of 30+ signals × 4 lags (0,1,2,3) │
   │   → Pearson correlation with Revenue        │
   │   → Keep if p < 0.20                        │
   │   → Rank by R²                              │
   │   → Pick top 7 non-collinear features       │
   │   → Fit Multivariate OLS                    │
   │   → Get coefficient per feature             │
   └─────────────────────────────────────────────┘
         │
         ▼
Step 5: Calculate ₹ Impact per signal
   Impact = (value_d2 - value_d1) × OLS_coefficient
         │
         ▼
Step 6: Print Markdown Table
   | Driver Category | Metric | Apr 19 | Apr 20 | Delta | Impact | Insight |
   | Competitor Pressure | Comp_Squeeze | 0.52 | 0.55 | +0.03 | ₹-78 | ... |
   | Own Levers | Organic_SOV | 4.1% | 5.2% | +1.1% | ₹-5,429 | ... |
   | Unexplained | Market Fluctuation | - | - | - | ₹-15,363 | ... |
```

### Signal Categories

| Category | Signals Tested | What It Means |
|----------|---------------|---------------|
| **Own Levers** | `Wt_OSA`, `Discount`, `Ad_SOV`, `Organic_SOV` | Things we directly control |
| **Competitor Pressure** | `Comp_Discount`, `Comp_OSA`, `Comp_Ad_SOV`, `Comp_Disc_Adv`, `Comp_Squeeze` | What competitors are doing |
| **Store Quality** | `HV_OSA`, `LV_OSA`, `HV_Stock`, `LV_Stock`, `Stock_roll3`, `Store_OSA`, `Stores`, `Listing` | How our stores are performing |
| **Structural** | `Is_Monday`, `DOW` | Day-of-week natural demand pattern |

---

## How to Read the Output

### Console Output (while running)

```
[LOAD] Reading File 1 ...           ← Data loading started
    File1 rows: 1,966,638           ← Store-level rows loaded
[L0] Raw daily snapshot ...         ← Layer 0 starting
    Apr 19  Sun  Rs.17,68,889       ← Raw numbers, trust these first
    Apr 20  Mon  Rs.13,78,235
[L4] Lag correlation ...            ← When does OSA affect revenue?
    OSA: lag=1d  r=0.508  p=0.026   ← OSA today → Revenue tomorrow, strong signal
[L6] Competitive separation ...
    Verdict: OWN_ISSUE              ← ⚠️ Category grew but we fell = our fault
[COMPARE] 2026-04-19 vs 2026-04-20 ← Deep dive starting
```

### Executive Summary MD Sections

```
## L0 — RAW DAILY SNAPSHOT
→ Start here. These are the verified numbers.
→ Look at the trend: is OSA dropping day after day?

## PRIMARY COMPARISON
→ The headline number. Revenue delta + % change.
→ Check "VERDICT" — OWN_ISSUE means internal problem.

### SKU DEEP DIVE
→ For each top-losing SKU, read city by city.
→ Each table row = one driver. Negative impact = it HURT sales.
→ Unexplained row at bottom = what we couldn't explain.

## L8 — LEADING INDICATOR ALERTS
→ Forward-looking. What will drop TOMORROW if not fixed?
→ [HIGH] = act today. [MEDIUM] = monitor closely.

## L10 — CONFIDENCE
→ Score < 60% = treat as hypothesis, not fact.
→ Always check "Reasons" before acting on findings.
```

### Reading the City Tables

```
| Driver Category  | Metric           | Apr 19 | Apr 20 | Delta  | Impact      | Insight          |
|------------------|------------------|--------|--------|--------|-------------|------------------|
| Own Levers       | Discount         | 31.7%  | 18.4%  | -13.3% | ₹-98,568    | coef: 7398/unit  |
| Competitor       | Comp_Disc_Adv    | -8.62  | 4.08   | +12.70 | ₹+79,633    | Price gap swing  |
| Store Quality    | HV_Stock         | 3.00   | 2.74   | -0.26  | ₹+3,310     | coef: -12908/unit|
| Unexplained      | Market Fluctuation | -    | -      | -      | ₹-847       | Natural noise    |

HOW TO READ:
─────────────
- Bold ₹ = NEGATIVE impact (this driver HURT revenue)
- Italic ₹ = POSITIVE impact (this driver HELPED revenue)
- Delta positive (+) = the metric increased from d1→d2
- "Lag 1d" in metric name = impact felt 1 day AFTER the change
- Unexplained = portion not statistically tied to any measured driver
- R² shown for each city = model accuracy (0.89 = 89% of drop explained)
```

---

## Output Files

After running, a timestamped folder is created:
```
output/
└── RCA_April 23-2026 01-00 PM/
    ├── advanced_rca_executive_summary.md  ← Main human-readable report
    ├── advanced_rca_report.json           ← Full machine-readable report
    ├── llm_evidence_pack.xlsx             ← 9-sheet Excel for deeper analysis
    └── llm_evidence_pack.json             ← JSON version of evidence pack
```

### Excel Sheets in `llm_evidence_pack.xlsx`

| Sheet | What's In It | Best For |
|-------|-------------|----------|
| `daily_national` | National daily metrics + category data | Trend analysis |
| `sku_city_day` | All 11K rows: every SKU × City × Day | Pivot analysis |
| `sku_day_summary` | SKU aggregated across cities per day | SKU ranking |
| `city_day_summary` | City aggregated across SKUs per day | City ranking |
| `sku_city_comparison` | d1 vs d2 snapshot per SKU × City | Impact sizing |
| `locality_intelligence` | High-value store stats per SKU × City | Store quality |
| `driver_attributions` | National-level ₹ attribution per driver | Finance team |
| `sku_city_attr_day` | Day-level causal attribution per SKU × City | Deep audit |
| `sku_city_attr_period` | Week-level causal attribution | Weekly review |

---

## Key Metrics Explained

| Metric | Source | What It Means | Good Value |
|--------|--------|--------------|-----------|
| **OSA %** | FILE2 (weighted) | % of time product was in stock and visible | > 75% |
| **Discount %** | FILE2 (weighted) | Average discount offered | 5–16% |
| **Ad SOV %** | FILE2 | Share of Blinkit ad spend vs category | < 28% |
| **Organic SOV %** | FILE2 | Organic search visibility share | Higher = better |
| **Network_Strength** | FILE1 → aggregated | Sum of Locality Sales Contribution across all active stores × 100 | Higher = better |
| **Darkstore_Count** | FILE1 → aggregated | Number of stores stocking this SKU | Higher = better |
| **HV_OSA** | FILE1 → grouped | Average OSA % in top 25% revenue stores | > 90% |
| **Comp_Disc_Adv** | FILE2 (competitors) | Competitor avg discount MINUS our discount (positive = they discount more) | Negative preferred |
| **Comp_Squeeze** | Derived | `Comp_Disc_Adv × (100 - Own_OSA) / 100` — dual pressure index | Near 0 preferred |
| **Stock_roll3** | FILE1 → rolling | 3-day rolling average stock level | Stable preferred |
| **Is_Monday** | Derived | 1 if Monday (naturally lower demand day) | Used to isolate structural drops |

---

## Thresholds Reference

```python
THR = {
    "OSA_critical":      65.0,   # Below this = collapse risk
    "OSA_warning":       75.0,   # Below this = warning zone
    "Discount_dimret":   20.0,   # Above this = diminishing returns
    "SOV_saturation":    40.0,   # Above this = ad spend wasted
    "Dark_metro_min":    30,     # Minimum stores expected in metros
    "Pullforward_ratio": 0.5,    # Spike:drop ratio to flag as pullforward
    "Anomaly_z":         2.0,    # Z-score to flag as true anomaly
    "Trend_days":        2,      # Days of consecutive decline to alert
}
```

### OSA Zone Bands

```
OSA Value    Zone       Implication
──────────── ────────── ─────────────────────────────────────────
> 75%        SAFE       Normal operations
65–75%       RISK       Sales drag starting, monitor daily
< 65%        CRITICAL   Revenue collapse risk, act immediately
```

---

## Common Questions

**Q: Why does revenue drop even when OSA looks okay?**
> OSA is a national average. A city like Mumbai could be at 55% while national average shows 70%. Use the City × SKU deep dive tables to find the local reality.

**Q: What does "Lag 2d" mean in a table row?**
> The change in that metric from 2 days ago is what's affecting today's revenue. Example: OSA dropped on Apr 18 → Revenue dropped on Apr 20 (2-day lag).

**Q: What's the difference between Unexplained and Organic/Market?**
> - **Unexplained** in city tables = variance the local regression model couldn't capture (noise)
> - **Organic/Market** in national L9 = revenue movement that happened despite all drivers being stable (true market pull/push)

**Q: Confidence is 40% — should I trust the results?**
> Treat them as hypotheses, not facts. Check L0 raw numbers first. The low confidence usually means not enough data days (< 30). Directional findings (which city, which SKU) are still reliable even at low confidence.

**Q: The "Others" city shows up — what is it?**
> Cities in the data that don't match the named metros (Mumbai, Delhi-NCR, etc.) are grouped as "Others". It's all Tier 2/3 cities combined.

**Q: How do I add a new signal to the city analysis?**
> In `sku_deep_dive()`, add your calculated column to the `ts` DataFrame before the `_fit_sku_city_regression_dynamic(ts)` call. The engine will automatically test it and include it if statistically significant.

**Q: How do I change which SKUs get deep-dived?**
> In `compare_two_days()`, find `for s in top_sku_losers[:5]:` and change `5` to however many you want.

---

## Diagram: How a Revenue Drop Gets Explained

```
Revenue dropped ₹3,90,654 (Apr 19 → Apr 20)
         │
         ▼
L6 checks: Category grew 4% on same day
         │
         └──► VERDICT = OWN_ISSUE (our problem, not market)
                  │
                  ▼
         L7: Which cities bled?
         ┌──────────────────────┐
         │ Others:   -₹1,27,280 │
         │ Delhi:    -₹65,334   │
         │ Bangalore:-₹49,573   │
         │ Hyderabad:-₹48,940   │
         └──────────────────────┘
                  │
                  ▼
         compare_two_days → Top 5 losing SKUs
         ┌─────────────────────────────────┐
         │ 1. Jaggery Powder 500g  -₹41,140│
         │ 2. Sonamasuri Rice 5kg  -₹30,407│
         │ 3. Moong Dal 500g       -₹21,940│
         └─────────────────────────────────┘
                  │
                  ▼ sku_deep_dive() for each
         Per SKU → Per City → 30 signals tested
                  │
                  ▼
         Dynamic OLS → ₹ impact per signal
                  │
                  ▼
         ┌────────────────────────────────────────────┐
         │ CITY: Bangalore | Drop: ₹-8,386            │
         │ R² = 0.89 | Signals Tested: 20             │
         │                                            │
         │ Discount: 31.7%→18.4% → Impact: ₹-98,568  │
         │ Comp_Disc_Adv: swing   → Impact: +₹79,633  │
         │ Stock (HV stores):     → Impact: ₹-2,325   │
         │ Unexplained:           → ₹-847             │
         └────────────────────────────────────────────┘
```

---

*Last updated: April 2026 | Engine version: 2.0*
