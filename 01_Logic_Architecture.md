# Advanced Sales RCA Engine — Logic Architecture
### 24 Mantra Organic | Blinkit Platform
**Version 2.0 | April 2026** — Now includes LLM Evidence Pack + Day × SKU × City Locality Intelligence

---

## What This System Does

Most RCA tools answer: **"What changed?"**
This engine answers: **"Why it changed, how drivers interact, will it repeat, and what to do next."**

It processes ~500MB of raw Blinkit data through 10 sequential analytical layers, each removing one type of noise or adding one layer of intelligence, until what remains is a precise, financially-quantified, forward-looking root cause analysis.

In addition, the engine now runs a **Locality Intelligence Engine** in parallel that operates at the most granular level possible — `(Date × SKU × City)` — and exports a structured **LLM Evidence Pack** (Excel + JSON) that you can feed directly into Claude/GPT for free-form analysis. This separates the engine into two roles: **Analyst** (narrative summary) and **Data Factory** (raw evidence for LLM).

---

## Full Pipeline Diagram

```
RAW DATA (2 CSV files, ~500MB)
│
│  File 1: blinkit-availability-data-april26-day wise.csv
│  └── Store-level: OSA %, Discount %, Listing %, MRP, SP
│      1,966,638 rows x 26 columns
│
│  File 2: blinkit-rca-download-April-26-Daily-City-Comp.csv
│  └── City-SKU-level: Offtake, SOV, Discount, OSA, Category Share
│      467,570 rows x 22 columns
│
▼
┌─────────────────────────────────────────────┐
│  DATA LOADING & DARKSTORE DERIVATION        │
│  Filter: Brand = "24 mantra organic"        │
│  Derive Darkstore Count from File 1         │
│  (rows where Avg. OSA % = 100 → 1 store)   │
└─────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  NATIONAL DAILY TIME-SERIES                 │
│  Aggregate to 1 row per date                │
│  Cols: Revenue, OSA, Discount, SOV,         │
│        Darkstores, Units                    │
└─────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  ┌────────┐      ┌────────┐
  │  L1    │      │  L4    │
  │Baseline│      │  Lag   │
  └───┬────┘      └───┬────┘
      │               │
      ▼               ▼
  ┌────────┐      ┌────────┐
  │  L3    │      │  L2    │
  │Thresh  │      │Interact│
  └───┬────┘      └───┬────┘
      └───────┬───────┘
              ▼
  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
  │  L5    │  │  L6    │  │  L7    │  │  L8    │
  │Pullfwd │  │Compet  │  │  Geo   │  │Leading │
  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘
      └───────────┴───────────┴───────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  L9: FINANCIAL ATTRIBUTION    │
              │  Rs. impact per driver        │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  L10: CONFIDENCE + FEEDBACK   │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  COMPARISON ENGINE            │
              │  Apr 19 vs Apr 20 deep dive   │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
  advanced_rca_report.json   advanced_rca_executive_summary.md
  (structured data, 54KB)    (narrative, human-readable)


  PARALLEL TRACK: LOCALITY INTELLIGENCE ENGINE (NEW in v2.0)
  ──────────────────────────────────────────────────────────

  active_stores (raw store-level data)
       │
       ▼
  build_locality_intelligence()
  → Per (Date, SKU, City): HHI, P75, Avg Score, High/Low store counts
       │
  build_sku_city_ts()
  → Joins with revenue/OSA/Discount per SKU x City, adds DoD columns
  → Adds City_Revenue_Share_Pct and City_Network_Share_Pct
       │
  export_llm_evidence_pack()
       │
       ├── llm_evidence_pack.xlsx  (7 sheets, ~2.8 MB)
       └── llm_evidence_pack.json  (3 keys, ~16 MB)
```

---

## Data Sources — Explained

### File 1: Availability Data (Store-Level)
Every row = one SKU in one store on one date.

| Column | Meaning |
|--------|---------|
| `Date` | The day |
| `Store ID` | Blinkit darkstore ID |
| `Product ID` | SKU identifier |
| `Locality City` | City (lowercase: "mumbai", "delhi-ncr") |
| `Avg. OSA %` | On-Shelf Availability — 100=in stock, 0=out |
| `Wt. Disc %` | Discount % being offered |
| `Listing %` | Is SKU listed in this store? |

**How Darkstore Count is derived:**
```
Darkstore Count = COUNT(rows where Avg. OSA % == 100)
                  GROUP BY (Date, City, Product ID)

Example:
Date        City    Product ID  Avg.OSA%
2026-04-19  mumbai  16658       100      ← counts = 1 darkstore
2026-04-19  mumbai  16658       100      ← counts = 1 darkstore
2026-04-19  mumbai  16658       0        ← does NOT count
─────────────────────────────────────────
Darkstore_Count (Product 16658, Mumbai, Apr19) = 2
```

---

### File 2: City-Level RCA Metrics
Every row = one SKU in one city on one date.

| Column | Meaning |
|--------|---------|
| `Offtake MRP` | Revenue at MRP |
| `Units` | Units sold |
| `Wt. OSA %` | Weighted avg OSA across all stores in that city |
| `Wt. Discount %` | Weighted avg discount |
| `Ad SOV` | Ad Share of Voice vs competitors |
| `Category Share` | Brand's share of category sales |

---

## Layer-by-Layer Architecture

---

### L1 — Smart Baseline

**Problem with simple average:**
A 20-day average as baseline creates false anomalies because sales trend
and have weekday/weekend patterns. A Sunday always looks like an anomaly
vs a Tuesday average baseline — but it's just weekend seasonality.

**What L1 removes:**

```
Raw Sales = Trend Component + Day-of-Week Effect + True Residual
                                                       ↑
                                          Only investigate THIS
```

**Math step by step:**
```
Step 1: Fit linear trend
        y_trend = LinearRegression(Revenue ~ DayNumber)
        → extracts underlying growth/decline curve

Step 2: Detrend
        y_detrend = Revenue - y_trend

Step 3: Day-of-Week seasonality
        For each day: Monday_avg, Tuesday_avg ... Sunday_avg
        DOW_effect[i] = mean(y_detrend) for days matching weekday[i]

Step 4: True residual
        Residual = y_detrend - DOW_effect

Step 5: Z-score
        Z = (Residual - mean) / std
        |Z| > 2.0 → TRUE anomaly (investigate)
        |Z| < 2.0 → Expected variation (ignore)
```

**Example from April data:**
```
Trend slope:  Rs.-4,989/day
→ Sales declining ~Rs.5K/day underlying throughout April

True anomaly dates: []
→ Zero true anomalies detected
→ The Apr20 drop was WITHIN expected variance of the declining trend
→ The trend itself is the problem — not a single-day event
```

**Output fields:**
```json
{
  "trend_slope_day": -4989.13,
  "trend_r2": 0.0466,
  "true_anomaly_dates": [],
  "dow_effects": {"0": 5200, "6": 18000}
}
```

---

### L4 — Lag Analysis

**Why lag matters:**
```
Action today         Effect visible
─────────────────────────────────────
OSA drops            Same day (customers can't buy what isn't there)
Discount removed     Same day (immediate price response)
Ad SOV cut           2-3 days later (awareness decays slowly)
Darkstore goes down  Next day (1 delivery cycle to feel impact)
```

If you blame today's OSA for today's drop — you may be 1 day off.
The real cause is yesterday's darkstore reduction.

**Math:**
```python
for lag in [0, 1, 2, 3]:
    correlation, p_value = pearsonr(driver.shift(lag), revenue)
best_lag = lag with highest |correlation|
```

**April actual results:**
```
Driver        Best Lag    r       p        Significant?
─────────────────────────────────────────────────────────
OSA           1 day       0.508   0.026    YES
Discount      0 days      0.455   0.044    YES
Ad_SOV        2 days      0.403   0.097    NO (need more data)
Darkstores    1 day       0.628   0.004    YES (strongest)
```

**Key insight:** Darkstores is the strongest driver (r=0.628).
A darkstore going offline Monday → sales drop Tuesday.
So when Apr20 sales dropped, the responsible darkstore count
is Apr19's count, not Apr20's.

---

### L3 — Threshold Logic

**Non-linear reality:**
```
OSA % vs Sales Impact (illustrative):

100% |████████████████  Normal sales
 80% |████████████████  Slight drag (-5%)
 75% |████████████      Warning zone (-15%)
 65% |████████          CLIFF EDGE — collapse begins (-30%)
 60% |████              Severe drop (-50%)
 50% |██                Near-zero possible

It is NOT a straight line. There's a threshold effect.
```

**Configured thresholds:**
```
OSA Critical:     65%   below = disproportionate collapse
OSA Warning:      75%   below = drag starting
Discount Dimret:  20%   above = customers wait for deals
SOV Saturation:   40%   above = marginal spend near-zero ROI
```

**April status:**
```
OSA:      69.27% → WARNING (between 65% critical and 75% warning)
Discount:  9.54% → OK (well below 20% cap)
Ad_SOV:    5.19% → MONITOR (far below 40% saturation)
```

**What WARNING means for OSA at 69.27%:**
The system is 4.27 percentage points above the critical cliff.
If OSA declines another 5% (which L8 shows is happening), it
crosses into collapse territory.

---

### L2 — Driver Interactions

**The most under-analysed phenomenon in RCA:**
Drivers are not independent. Their combination matters more
than their individual values.

**Interaction matrix:**

```
           OSA High     OSA Low
           ─────────    ──────────────────────────
SOV High   AMPLIFIED    WASTAGE — fix OSA first
           sales win    (ads seen, product not there)

SOV Low    Needs more   DOUBLE MISS — both broken
           investment   
```

**Rules implemented with examples:**

**Rule 1: SOV x OSA Wastage**
```
Condition: Ad_SOV > 5% AND OSA < 75%
Example:   SOV = 5.2%, OSA = 69.3%
Verdict:   Every ad rupee is creating awareness but
           customers arrive and product is missing.
Action:    PAUSE ad spend, FIX OSA first, THEN resume ads.
```

**Rule 2: Double Penalty**
```
Condition: OSA < 65% AND Discount < 5%
Example:   OSA = 60%, Discount = 2%
Verdict:   Product unavailable + no price incentive = maximum drag
Action:    Emergency OSA fix + consider tactical discount
           to compensate for availability gap
```

**Rule 3: Discount x Dark Reach**
```
Condition: Discount > 10% AND Darkstores < 30
Example:   Discount = 12%, Darkstores = 15
Verdict:   Promotion is running but only 15 stores can deliver.
           Most customers get promoted to a product that
           is out of stock when they try to order.
Action:    Fix supply before next promotional wave
```

**April finding:**
```
[ACTIVE WARNING] SOV x OSA Wastage
Ad SOV 5.2% running + OSA 69.3% (below warning threshold)
→ SOV spend is being wasted right now
```

---

### L5 — Pullforward & Cannibalization Filter

**The pattern:**
```
Day 1 (discount):   +40% sales spike    ← looks great
Day 2 (no disc):    -25% sales drop     ← looks like crisis

Basic RCA: Investigates Day 2 as an anomaly
Advanced:  Recognizes Day 2 is echo of Day 1's pullforward
```

**Decision rule:**
```
spike_pct = (Day1_sales - Day0_sales) / Day0_sales
drop_pct  = (Day1_sales - Day2_sales) / Day1_sales

IF drop_pct ≤ spike_pct × 0.5  →  PULLFORWARD (not a real problem)
IF drop_pct > spike_pct × 0.5  →  REAL DROP (investigate)

Example:
Spike: +40% on Monday
Drop:  -18% on Tuesday
Test:  0.18 ≤ 0.40 × 0.5 = 0.18 ≤ 0.20  → PULLFORWARD
Result: Tuesday drop is expected, not worth investigating
```

**April result:**
0 pullforward events (discount was only 9-10%, not enough to trigger stocking behaviour)

---

### L6 — Competitive & Market Separation

**The most important first question:**
Before investigating any lever, ask: was this our problem or the market's?

```
Scenario                    Right action
────────────────────────────────────────────────────────────────
Your brand fell, market    Macro headwind — do NOT change levers,
also fell equally          wait it out

Your brand fell, market    YOUR problem — levers broken, fix them
was flat or rose

Your brand rose, market    You're winning — double down on what
also rose                  is working
```

**April 19→20 verdict:**
```
Brand fell:    -22.1%
Category fell: -22.1%  ← exactly matched

Verdict: MARKET_ISSUE
Interpretation: Blinkit as a platform had lower demand on Apr20
(possibly a Monday effect, or platform-wide category slowdown).
24 Mantra Organic moved in lockstep with the market.
Brand share unchanged at ~100% of organic category.

Action: Do NOT react with discounts or ad spend increase.
        Monitor for 2-3 more days to see if market recovers.
```

---

### L7 — Geographic Cascade

**National numbers hide regional stories:**
```
National: -22%
    ↓ decompose
├── Others (Tier2/3): -29%  ← Rs.-1.27L  (33% of total drop)
├── Delhi-NCR:        -28%  ← Rs.-0.65L
├── Bangalore:        -19%  ← Rs.-0.49L
├── Hyderabad:        -22%  ← Rs.-0.49L
└── Mumbai:           -18%  ← Rs.-0.42L
```

**Two patterns and interpretation:**
```
CONCENTRATED (top 2 cities > 70% of drop):
  → Likely OPERATIONAL — specific darkstore outage, OSA crash
  → Fix: Emergency restocking in those 2 cities specifically

SPREAD (drop distributed across many cities):
  → Likely MACRO — platform issue, competitor, category decline
  → Fix: National-level response, not city-specific firefighting
```

**April pattern: SPREAD**
```
Others + Delhi = Rs.-1.92L = 49% of total — below 70% threshold
→ SPREAD confirmed → MACRO issue → aligns with MARKET_ISSUE verdict from L6
```

---

### L8 — Leading Indicator Engine

**Why forward-looking matters:**
By the time you detect a problem in RCA, it already happened.
L8 flags where the NEXT problem is building.

**Logic:**
```
IF any driver declines for N consecutive days (N=2 in config):
  → Raise alert: "Sales at risk in 2-3 days"

IF OSA in a specific city has declined 2+ days:
  → City-specific alert
```

**April alerts raised (12 total):**
```
[HIGH] Darkstore Count: -101 over 2 days
       → Sales at risk nationally in next 2-3 days

[HIGH] Mumbai OSA: now 63.7% (declining 2 days)
       → Below 65% critical threshold — URGENT

[HIGH] Lucknow OSA: 63.4% (below critical)
[HIGH] Others OSA: 64.6% (near critical)
[HIGH] Bangalore, Chennai, Hyderabad, Pune — all declining
```

**This is the forward prediction the engine makes:**
If these trends continue even one more day, sales on Apr21 and Apr22
will see further drops driven by OSA crossing the 65% critical threshold
in multiple cities simultaneously.

---

### L9 — Financial Attribution

**From "driver rankings" to "rupee impact":**
```
Basic output:  "Darkstores is most correlated"
L9 output:     "Darkstores caused Rs.-1,23,000 of the Rs.-3,90,654 drop"
```

**Method — correlation-weighted allocation:**
```
1. Total delta = Rs.-3,90,654

2. Correlation strengths (from L4):
   Darkstores: r = 0.628  → weight = 0.628/(0.628+0.508+0.455+0.403) = 0.315
   OSA:        r = 0.508  → weight = 0.255
   Discount:   r = 0.455  → weight = 0.228
   Ad_SOV:     r = 0.403  → weight = 0.202

3. Revenue impact:
   Darkstores: Rs.-3,90,654 × 0.315 = Rs.-1,23,000
   OSA:        Rs.-3,90,654 × 0.255 = Rs.-0,99,600
   Discount:   Rs.-3,90,654 × 0.228 = Rs.-0,89,200
   Ad_SOV:     Rs.-3,90,654 × 0.202 = Rs.-0,78,900
```

**Business use of these numbers:**
```
"If we restore Darkstores by 101 stores and OSA back to 70%+,
 potential revenue recovery = Rs.-1,23,000 + Rs.-99,600 = Rs.-2,22,600/day"

Share this with:
- Supply Chain: Rs.1.23L is the cost of 101 darkstores going offline
- Marketing: Rs.99K is the OSA-driven loss (not ad spend issue)
- Finance: Total recoverable = Rs.2.2L of the Rs.3.9L drop
```

---

### L10 — Confidence Score

**Formula:**
```
Confidence = (trend_r2 × 30) + (top_driver_r × 40) + (data_days/30 × 20) + bonus

April:
= (0.047 × 30) + (0.628 × 40) + (20/30 × 20) + 5
= 1.4 + 25.1 + 13.3 + 5
= 44%  → LOW
```

**What each component measures:**
| Component | Weight | Measures |
|-----------|--------|---------|
| trend_r2 | 30 | How well we modelled the baseline |
| top_driver_r | 40 | How strong is our best driver signal |
| data_days | 20 | Do we have enough historical data |
| anomaly bonus | 10 | Did we actually detect anomalies |

**Why 44% is actually OK for Apr data:**
- Only 20 days — statistical confidence needs 60+
- But directional findings (MARKET_ISSUE, SPREAD, OSA WARNING) are still valid
- The correlation magnitudes and thresholds are correct
- Treat as: "High confidence in direction, lower confidence in exact Rs. numbers"

---

## NEW (v2.0) — Locality Intelligence Engine

### What Problem This Solves

Before v2.0, `Network_Strength` was a **national aggregate** — one number per day. The executive summary could say "You gained 67 stores" but could not tell you:

- Were those 67 stores high-value (large, busy localities) or low-value (small, thin stores)?
- For **Jaggery 500g in Bangalore specifically**, did the store quality drop?
- Is our network in Delhi-NCR concentrated into 1-2 massive stores (high risk) or spread across many (low risk)?

Now the engine computes all of this at the most granular level: **(Date, SKU, City)**.

---

### The `Locality Sales Contribution` Column

This is the core signal — a **store-level importance weight** from File 1 (`Locality Sales Contribution`).

```
What it represents:
  A decimal fraction representing how much sales volume
  a store's locality typically contributes.

  Example values:
    0.00082  → low-volume locality (thin store, small area)
    0.00168  → HIGH-VALUE threshold (defined in engine config)
    0.00320  → premium locality (busy, large catchment area)
    0.00891  → flagship location

Note: This column comes from File 1 (availability data).
Only rows where Avg. OSA % == 100 are included
(i.e., only ACTIVE, fully-stocked stores count).
```

---

### `build_locality_intelligence()` — What It Computes

**Input:** `active_stores` dataframe (File 1, filtered to OSA=100)
**Grain:** One row per (Date, city_key, Product ID)

| New Column | Formula | Business Meaning |
|---|---|---|
| `Darkstore_Count` | COUNT(Store ID) per group | Raw active store count |
| `Avg_Locality_Score` | MEAN(Locality Sales Contribution) | Are your active stores high or low quality? |
| `Max_Locality_Score` | MAX(Locality Sales Contribution) | Your single best store in that city for that SKU |
| `Min_Locality_Score` | MIN(Locality Sales Contribution) | Your weakest active store |
| `P75_Locality_Score` | 75th PERCENTILE(Locality Sales Contribution) | Where most of your network mass sits |
| `High_Value_Store_Count` | COUNT(stores where score ≥ 0.00168) | Stores that actually drive meaningful volume |
| `Low_Value_Store_Count` | Darkstore_Count − High_Value_Store_Count | Stores that add count but little sales |
| `High_Value_Store_Pct` | High_Value / Total × 100 | Network quality ratio |
| `Network_Strength` (raw) | SUM(Locality Sales Contribution) | Pre-scaled total capacity |
| `HHI` | Σ(share_i²) where share = score/total×100 | Concentration index (see below) |

**HHI — Herfindahl-Hirschman Index:**
```
HHI measures whether your network is
spread evenly or concentrated in a few stores.

  HHI close to 0    →  Very spread (many equal stores)
                       Low risk — losing 1 store doesn't matter much

  HHI close to 10000 → Extremely concentrated (1-2 giant stores)
                       High risk — losing 1 store wipes your capacity

Example:
  Bangalore: 5 stores, scores = [0.003, 0.003, 0.003, 0.003, 0.003]
  HHI = 5 × (20%)² = 2,000  → Low concentration, healthy spread

  Mumbai: 3 stores, scores = [0.008, 0.001, 0.001]
  HHI = (80%)² + (10%)² + (10%)² = 6,600  → High concentration, risky
  Losing the 0.008 store = -80% of network capacity overnight
```

---

### `build_sku_city_ts()` — The Master Granular Table

**Input:** df24 (File 2) + loc_stats from above
**Grain:** One row per (Date, Product ID, City)
**~16,500 rows** for 20 days × 75 SKUs × 11 cities

This table is the spine of the LLM Evidence Pack. Every row contains:

```
Identifiers:
  Date, Product ID, Product Name, Grammage, City, city_key, Tier

Performance Metrics:
  Revenue           — Offtake MRP (sum across SKU × City × Date)
  OSA               — Weighted avg OSA in that city for that SKU
  Discount          — Weighted avg discount
  Ad_SOV            — Average Ad SOV

Locality Intelligence:
  Darkstore_Count       — Active stores (OSA=100%)
  Network_Strength      — Scaled capacity (×100)
  Avg_Locality_Score    — Mean store quality
  Max_Locality_Score    — Best store quality in this city
  Min_Locality_Score    — Worst store quality
  P75_Locality_Score    — 75th percentile store quality
  HHI                   — Concentration index
  High_Value_Store_Count — Stores with score ≥ 0.00168
  Low_Value_Store_Count  — Stores below threshold
  High_Value_Store_Pct   — Quality ratio %

Day-over-Day Changes (vs prior day, same SKU × City):
  Revenue_DoD             — Revenue change from yesterday
  Network_Strength_DoD    — Network capacity change
  Darkstore_Count_DoD     — Raw store count change
  Avg_Locality_Score_DoD  — Did average store quality shift?

Market Share Within SKU (vs national total on same day):
  City_Revenue_Share_Pct  — % of this SKU's national revenue from this city
  City_Network_Share_Pct  — % of this SKU's national network from this city
```

**Why DoD columns matter:**
```
Scenario: Jaggery 500g, Bangalore, Apr19 vs Apr20

  Darkstore_Count:         12 → 9      (DoD = -3)
  High_Value_Store_Count:   5 → 2      (DoD = -3)
  Low_Value_Store_Count:    7 → 7      (DoD = 0)
  Avg_Locality_Score_DoD:  -0.00041

Conclusion:
  We lost 3 stores. ALL 3 were high-value stores.
  The low-value stores stayed. The network shrank in QUALITY, not just quantity.
  This is much more damaging than losing 3 low-value stores.
```

**Why City_Network_Share_Pct matters:**
```
Jaggery 500g, Apr 20:
  Delhi-NCR:   City_Network_Share_Pct = 28.4%
  Bangalore:   City_Network_Share_Pct = 22.1%
  Mumbai:      City_Network_Share_Pct = 6.8%

Interpretation:
  28% of Jaggery's entire network capacity sits in Delhi-NCR.
  If Delhi-NCR has a bad OSA day, it affects 28% of the SKU's sales capacity.
  Mumbai is only 6.8% — OSA problems there are less impactful for this SKU.
```

---

### `export_llm_evidence_pack()` — The 7-Sheet Output

**Output 1: `llm_evidence_pack.xlsx`** (~2.8 MB)

| Sheet | Grain | Key Purpose |
|---|---|---|
| `daily_national` | Date | 20 rows — top-level daily summary with category context |
| `sku_city_day` | Date × SKU × City | The master table (~16,500 rows) — all metrics + locality intelligence + DoD |
| `sku_day_summary` | Date × SKU | SKU performance collapsed nationally per day |
| `city_day_summary` | Date × City | City performance collapsed across SKUs per day |
| `sku_city_comparison` | SKU × City (d1 vs d2) | Side-by-side comparison for the two comparison dates with all deltas |
| `locality_intelligence` | Date × SKU × City | Only the locality-specific columns for focused store quality analysis |
| `driver_attributions` | One row per driver | Flat version of the financial attribution (OSA impact Rs., Discount impact Rs., etc.) |

**Output 2: `llm_evidence_pack.json`** (~16 MB)

Contains 3 top-level keys:
```json
{
  "meta": {
    "llm_prompt_hint": "Analyze the sku_city_day and locality_intelligence data ..."
  },
  "daily_national":       [ ...20 rows... ],
  "sku_city_day":         [ ...~16500 rows... ],
  "locality_intelligence":[ ...~16500 rows... ]
}
```

---

### Example Questions Now Answerable from Evidence Pack

```
Q: For Jaggery 500g in Bangalore, between Apr 19 and 20 — did we lose
   high-value or low-value stores? What was the quality shift?

→ Filter sku_city_day: Product Name = 'Jaggery 500g', City = 'Bangalore'
   Apr 19: High_Value_Store_Count = 5, Avg_Locality_Score = 0.00221
   Apr 20: High_Value_Store_Count = 2, Avg_Locality_Score = 0.00168
   Answer: We lost 3 high-value stores. The remaining stores are at the threshold.

Q: Which SKUs have High_Value_Store_Pct < 30%? These are under-penetrated.
→ Filter sku_city_day on latest date: High_Value_Store_Pct < 30
   → Identifies SKUs where most stores are low-quality (quantity without quality)

Q: Show me SKUs where Network_Strength_DoD is negative but Revenue_DoD is positive.
→ These are resilient SKUs — sales held up despite network shrinking.
   Understand WHY and protect whatever kept them afloat.

Q: Which cities have a City_Network_Share_Pct > 20% for Rice 5kg?
→ These cities are critical to Rice's national capacity.
   OSA drops here will materially impact national revenue.

Q: For Sugar 1kg, is the HHI increasing over time in Mumbai?
→ Increasing HHI = growing concentration = growing fragility risk.
```

---

## Thresholds Reference

| Driver | Critical | Warning | Action at breach |
|--------|----------|---------|-----------------|
| OSA | <65% | <75% | Emergency restock — every hour matters |
| Discount | >20% | >16% | Reduce discount — base sales eroding |
| Ad SOV | >40% | >28% | Reduce ad spend — marginal ROI |
| Darkstores (metro) | <30 | <50 | Ops escalation — delivery promise broken |

---

## Key Formulas

```
Darkstore Count  = COUNT(File1 rows, Avg.OSA% == 100) GROUP BY Date+City+SKU

Trend            = LinearRegression(Revenue ~ DayNumber)
DOW Effect       = mean(detrended_revenue) per weekday
True Residual    = Revenue - Trend - DOW_Effect
True Anomaly     = |zscore(Residual)| > 2.0

Best Lag         = argmax |pearsonr(driver.shift(lag), revenue)| for lag in [0,1,2,3]

Pullforward Test = (post_spike_drop_pct <= spike_pct × 0.5)

Market Verdict:
  MARKET_ISSUE   if brand_dod<0 AND cat_dod<0 AND abs(cat_dod) >= abs(brand_dod)×0.7
  OWN_ISSUE      if brand_dod < cat_dod
  OUTPERFORMING  otherwise

Attribution Wt   = |pearsonr(driver)| / sum(all |pearsonr|)
Revenue Impact   = total_delta × weight × direction_sign

Confidence Score = (r2×30) + (top_r×40) + (days/30×20) + (10 or 5)
```
