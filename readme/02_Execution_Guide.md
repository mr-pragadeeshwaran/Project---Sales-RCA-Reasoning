# Advanced Sales RCA Engine — Execution Guide
### 24 Mantra Organic | Blinkit Platform
**Version 2.0 | April 2026** — Includes LLM Evidence Pack

---

## Prerequisites

### Python Version
Requires Python 3.10 or higher.

Check your version:
```
python --version
```

### Required Libraries
Install all dependencies:
```
pip install pandas numpy scipy scikit-learn openpyxl
```

Verify installation:
```python
python -c "import pandas, numpy, scipy, sklearn, openpyxl; print('All OK')"
```

> **Note:** `openpyxl` is required for the new LLM Evidence Pack Excel export. Without it, the engine will fail during the export step.

---

## File & Folder Structure

```
Project - Sales RCA Reasoning\
│
├── advanced_rca_engine.py       ← MAIN ENGINE — run this
│
├── _append_layers.py            ← helper module (locality intelligence functions)
├── _part2.py                    ← helper (ignore)
├── _part3.py                    ← helper (ignore)
├── _part4.py                    ← helper (ignore)
│
└── output\                      ← ALL outputs land here
    └── RCA_April 22-2026 12-09 PM\   ← timestamped run folder
        ├── advanced_rca_report.json            (54 KB  — structured 10-layer analysis)
        ├── advanced_rca_executive_summary.md  (14 KB  — narrative, human-readable)
        ├── llm_evidence_pack.xlsx             (2.8 MB — 7-sheet granular evidence)
        └── llm_evidence_pack.json             (16 MB  — LLM-ready structured data)
```

**Source data files (NOT in project folder, referenced by path):**
```
D:\2. Area\24 Mantra\April-26\Sales RCA\
├── blinkit-availability-data-april26-day wise.csv     (492 MB)
└── blinkit-rca-download-April-26-Daily-City-Comp.csv  (100 MB)
```

---

## How to Run

### Step 1 — Open Terminal / Command Prompt

Navigate to the project folder:
```
cd "c:\Users\cpsge\.gemini\antigravity\scratch\Project - Sales RCA Reasoning"
```

### Step 2 — Run the Engine
```
python advanced_rca_engine.py
```

### Step 3 — Watch the Progress Output
```
[OUTPUT] Folder: ...\output\RCA_April 22-2026 12-09 PM

[LOAD] Reading File 1 (availability) ...
[LOAD] Reading File 2 (RCA metrics) ...
    File1 rows: 1,966,638  |  File2 (24M): 11,415  |  All brands: 467,570

Date range: 2026-04-01 to 2026-04-20  | 20 days

[LLM Evidence] Building locality intelligence from raw stores ...   ← NEW
[LLM Evidence] Building SKU x City time-series ...                  ← NEW
[L0] Raw daily snapshot (last 7 days) ...
[L1] Baseline decomposition ...
    Slope: Rs.-4989.13/day | Anomalies: 0
[L4] Lag correlation ...
    OSA: lag=1d  r=0.508  p=0.026
    Discount: lag=0d  r=0.455  p=0.044
    Ad_SOV: lag=2d  r=0.403  p=0.097
    Network_Strength: lag=1d  r=0.417  p=0.076
[L3] Thresholds ...
    OSA: 69.27 -> WARNING
    Discount: 9.54 -> OK
    Ad_SOV: 5.19 -> MONITOR
[L2] Driver interactions ...
    Active: 1  Warnings: 1
[L5] Pullforward filter ...
    Pullforward: 0  Real: 0
[L6] Competitive separation (correct category back-calc) ...
    Verdict: OWN_ISSUE  Brand: -22.1%  Category: 4.0%
[L7] Geographic cascade ...
    Geo pattern: SPREAD (effective: SPREAD_OWN)
[L8] Leading indicators ...
    Alerts raised: 12
[L9] Financial attribution ...
    Delta: Rs.-390654  Top: OSA
[L10] Confidence scoring ...
    Confidence: 40%
[COMPARE] 2026-04-19 vs 2026-04-20 deep-dive ...
[LLM Evidence] Exporting evidence pack ...                          ← NEW
    -> llm_evidence_pack.xlsx
    -> llm_evidence_pack.json

[OK] JSON report  -> ...\advanced_rca_report.json
[OK] Executive Summary -> ...\advanced_rca_executive_summary.md
```

### Step 4 — See the Comparison Narrative in Console
```
=== 24 Mantra Organic | 2026-04-19 vs 2026-04-20 ===

HEADLINE: Revenue DECLINE of 22.1% (Rs.-390,654)
...
```

**Approximate runtime breakdown:**

| Phase | What happens | Time |
|---|---|---|
| Data loading | Reading File 1 (1.9M rows) + File 2 (467K rows) | 20–30 sec |
| Locality Intelligence | Groupby + P75 + HHI per SKU × City | 15–25 sec |
| L1–L10 Analytics | Baseline, lag, geo, leading alerts, attribution | 5–10 sec |
| Comparison Engine | SKU deep dives for top 5 losers | 5–10 sec |
| LLM Evidence Pack export | Writing 16MB JSON + 2.8MB Excel | 10–20 sec |
| **Total** | | **55–95 seconds** |

> Most of the time is spent on disk I/O (reading the CSVs) and the HHI/P75 computations across 16,500 (Date × SKU × City) combinations. This is a one-time cost per run.

Old run time (v1.0): ~35–50 seconds — the new locality intelligence adds ~20–40 seconds.

---

## How to Change the Comparison Dates

Open `advanced_rca_engine.py` and find these two lines near the top (around line 25):

```python
COMPARE_DATE1 = "2026-04-19"
COMPARE_DATE2 = "2026-04-20"
```

Change to any two dates within your data range (Apr 1 – Apr 20):

**Examples:**
```python
# Compare first week vs second week best day
COMPARE_DATE1 = "2026-04-05"
COMPARE_DATE2 = "2026-04-12"

# Compare a Saturday vs Sunday
COMPARE_DATE1 = "2026-04-18"
COMPARE_DATE2 = "2026-04-19"
```

Then run again:
```
python advanced_rca_engine.py
```

A new timestamped folder will be created. Old folders are preserved.

---

## How to Change Thresholds

Find the `THR` dictionary near the top of the engine file (~line 28):

```python
THR = {
    "OSA_critical":      65.0,   # % — below this, sales collapse
    "OSA_warning":       75.0,   # % — below this, drag starting
    "Discount_dimret":   20.0,   # % — above this, diminishing returns
    "SOV_saturation":    40.0,   # % — above this, marginal spend wasted
    "Dark_metro_min":    30,     # stores — below this, delivery broken
    "Pullforward_ratio": 0.5,    # post-spike drop ratio for pullforward tag
    "Anomaly_z":         2.0,    # z-score for true anomaly detection
    "Trend_days":        2,      # consecutive declining days for alert
}
```

Adjust based on your business knowledge. Example:
```python
# If your category's OSA critical point is actually 70%, change:
"OSA_critical": 70.0,
"OSA_warning":  80.0,
```

---

## Understanding the Two Output Files

### Output 1: `advanced_rca_report.json`

Full structured data from all 10 layers. Machine-readable.

**Top-level structure:**
```json
{
  "meta": { "brand": "24 Mantra Organic", "date_range": "...", "total_days": 20 },
  "L1_baseline":     { "trend_slope_day": -4989, "true_anomaly_dates": [] },
  "L2_interactions": { "warnings": [...], "positives": [...] },
  "L3_thresholds":   { "OSA": { "status": "WARNING", "current": 69.27 } },
  "L4_lag":          { "OSA": { "best_lag_days": 1, "best_r": 0.508 } },
  "L5_pullforward":  { "pullforward_events": [], "real_drops": [] },
  "L6_competitive":  { "latest_verdict": "MARKET_ISSUE" },
  "L7_geographic":   { "top_losing_cities": [...], "drop_pattern": "SPREAD" },
  "L8_leading":      { "alerts": [...], "total_alerts": 12 },
  "L9_attribution":  { "driver_attributions": [...], "top_driver": "Darkstores" },
  "L10_feedback":    { "confidence_score": 44, "interpretation": "LOW" },
  "COMPARISON": {
    "comparison_dates": { "d1": "2026-04-19", "d2": "2026-04-20" },
    "headline_delta_rs": -390654,
    "market_verdict": "MARKET_ISSUE",
    "driver_attribution": [...],
    "city_detail": [...],
    "top_sku_losers": [...],
    "narrative": "=== 24 Mantra Organic | ..."
  }
}
```

**When to use JSON:**
- Building a dashboard in Power BI / Tableau / Streamlit
- Feeding into another Python analysis
- Storing historical RCA records in a database
- When you need exact city × SKU breakdown numbers

---

## The Four Output Files

Every run produces 4 files in the timestamped output folder:

### Output 1: `advanced_rca_report.json` (54 KB)

Full structured data from all 10 layers. Machine-readable, deeply nested.

**Top-level structure:**
```json
{
  "meta": { "brand": "24 Mantra Organic", "date_range": "...", "total_days": 20 },
  "L0_raw_snapshot":  { "last_n_days": [...], "dod_changes": {...} },
  "L1_baseline":      { "trend_slope_day": -4989, "true_anomaly_dates": [] },
  "L2_interactions":  { "warnings": [...], "positives": [...] },
  "L3_thresholds":    { "OSA": { "status": "WARNING", "current": 69.27 } },
  "L4_lag":           { "OSA": { "best_lag_days": 1, "best_r": 0.508 } },
  "L5_pullforward":   { "pullforward_events": [], "real_drops": [] },
  "L6_competitive":   { "latest_verdict": "OWN_ISSUE" },
  "L7_geographic":    { "top_losing_cities": [...], "drop_pattern": "SPREAD_OWN" },
  "L8_leading":       { "alerts": [...], "total_alerts": 12 },
  "L9_attribution":   { "driver_attributions": [...], "top_driver": "OSA" },
  "L10_feedback":     { "confidence_score": 40, "interpretation": "LOW" },
  "COMPARISON": {
    "comparison_dates": { "d1": "2026-04-19", "d2": "2026-04-20" },
    "headline_delta_rs": -390654,
    "market_verdict": "OWN_ISSUE",
    "driver_attribution": [...],
    "city_detail": [...],
    "top_sku_losers": [...],
    "sku_deep_dives": [...]
  }
}
```
**Use when:** Building dashboards, historical comparisons, or programmatic access.

---

### Output 2: `advanced_rca_executive_summary.md` (14 KB)

Pre-written narrative covering all 10 layers. Human-readable Markdown.

**Structure:**
```
## L0 -- RAW DAILY SNAPSHOT (Evidence Base)
## PRIMARY COMPARISON: Apr 19 vs Apr 20
## L4 -- LAG ANALYSIS
## L3 -- THRESHOLD ZONES
## L2 -- DRIVER INTERACTIONS
## L1 -- SMART BASELINE
## L6 -- MARKET vs OWN FACTOR
## L7 -- GEOGRAPHIC CASCADE
## L8 -- LEADING INDICATOR ALERTS
## L9 -- FINANCIAL ATTRIBUTION
## L10 -- CONFIDENCE & FEEDBACK
```
**Use when:** Quick morning briefing, sharing with manager, getting LLM to draft summary email.

---

### Output 3: `llm_evidence_pack.xlsx` (2.8 MB) — NEW in v2.0

7-sheet Excel workbook. Every number the engine computed, in tabular form.

| Sheet | Rows | Use for |
|---|---|---|
| `daily_national` | 20 | National daily trends with category context |
| `sku_city_day` | ~16,500 | **Main analysis sheet** — full Day × SKU × City granularity with locality intelligence |
| `sku_day_summary` | ~1,500 | SKU performance by day (nationally aggregated) |
| `city_day_summary` | ~220 | City revenue and network per day |
| `sku_city_comparison` | ~800 | d1 vs d2 comparison at SKU × City level |
| `locality_intelligence` | ~16,500 | Store quality metrics only (HHI, P75, High/Low Value) |
| `driver_attributions` | 4 | Rs. impact per driver (OSA, Discount, SOV, Network) |

**Use when:** Filtering in Excel, browsing specific SKU + City combinations, copying data into Claude.

---

### Output 4: `llm_evidence_pack.json` (16 MB) — NEW in v2.0

Same as Excel but in JSON. Contains `daily_national`, `sku_city_day`, and `locality_intelligence` keys.
Includes a `meta.llm_prompt_hint` field with a suggested starting prompt.

**Use when:** Programmatic access, feeding into a custom LLM pipeline, or if Excel is too large for Claude's context window (use specific sheets/filters instead).

---

## How to Use with Claude

### Option A: Using the Executive Summary (Quick Analysis)

1. Open `output/RCA_.../advanced_rca_executive_summary.md`
2. Select All → Copy
3. Paste into Claude with your question:

```
[PASTE CONTENT]

My questions:
1. Why did sales drop on April 20?
2. Is this our fault or the market?
3. What should our supply chain team do tomorrow morning?
```

---

### Option B: Using the LLM Evidence Pack (Deep Analysis) — NEW in v2.0

This is for when you want Claude to answer very specific questions about a
particular SKU, city, or locality pattern — rather than the national narrative.

#### Step 1: Open the Excel file
```
output\RCA_...\llm_evidence_pack.xlsx
```

#### Step 2: Choose the right sheet

| Your question | Sheet to use |
|---|---|
| National trend / market context | `daily_national` |
| Which SKU + City lost high-value stores? | `locality_intelligence` |
| Full detail for one SKU across all cities | `sku_city_day` (filter by Product Name) |
| Apr 19 vs Apr 20 breakdown per SKU + City | `sku_city_comparison` |
| Revenue impact of OSA/Discount/Network | `driver_attributions` |

#### Step 3: Copy the relevant rows
Filter the sheet to your SKU / City / Date range.
Select All Visible → Copy.

#### Step 4: Paste into Claude with a specific question

**Example 1 — Store quality analysis:**
```
Here is the locality_intelligence data for Jaggery 500g in Bangalore
for April 19 and April 20:

[PASTE ROWS]

Questions:
- Did we lose high-value or low-value stores between Apr 19 and Apr 20?
- Is the HHI concentration increasing (fragility risk)?
- What does the P75_Locality_Score trend tell us?
```

**Example 2 — City risk analysis:**
```
Here is the sku_city_day data for all SKUs in Mumbai for the last 7 days:

[PASTE ROWS]

Questions:
- Which SKUs have City_Network_Share_Pct > 25% in Mumbai?
  These are the most exposed SKUs if Mumbai has an OSA failure.
- Show me SKUs where Avg_Locality_Score_DoD is negative for 3+ consecutive days.
```

**Example 3 — Network quality health check:**
```
Here is the locality_intelligence data for April 20 (latest day):

[PASTE ROWS]

Questions:
- Which SKU + City combinations have High_Value_Store_Pct < 30%?
  These are priority opportunities to expand into better localities.
- Rank SKUs by HHI concentration risk — highest HHI = highest risk.
```

**Example 4 — Resilient SKU identification:**
```
Here is the sku_city_day data for the comparison date:

[PASTE ROWS]

Find SKUs where Network_Strength_DoD is negative but Revenue_DoD is positive.
These are resilient SKUs. Explain what might explain the resilience
based on OSA, Discount, and Ad_SOV columns.
```

---

## Reading the Comparison Output — Example Walk-Through

**Scenario:** You ran the engine for Apr 19 vs Apr 20. Here is how to read the console output:

```
HEADLINE: Revenue DECLINE of 22.1% (Rs.-390,654)
  2026-04-19: Rs.1,768,889  |  2026-04-20: Rs.1,378,235
```
→ Total revenue fell by Rs.3.9 lakh in one day, a 22% decline.

```
MARKET CONTEXT:
  Blinkit category revenue fell 22.1% on same day.
  Verdict: MARKET_ISSUE
  => The overall category declined. Not a 24M-specific failure.
```
→ The entire organic category on Blinkit fell at the same rate.
   This is a platform/market issue, NOT a 24 Mantra Organic operational problem.
   Do not react by changing your levers.

```
DRIVER SNAPSHOT:
  OSA:        69.55 -> 69.27  (v -0.4%)
  Discount:   10.43 -> 9.54   (v -8.6%)
  Darkstores: 27858 -> 27757  (v -0.4%)
```
→ All drivers declined slightly, but none crashed dramatically.
   The decline was broad-based and small in magnitude.
   This confirms the drop is macro (market), not a single driver failure.

```
FINANCIAL ATTRIBUTION:
  Darkstores: Rs.-122,983  (31.5% of delta)  lag=1d
  OSA:        Rs. -99,593  (25.5% of delta)  lag=1d
  Discount:   Rs. -89,152  (22.8% of delta)  lag=0d
  Ad_SOV:     Rs. -78,926  (20.2% of delta)  lag=2d
```
→ The Rs.3.9L drop is attributed across all 4 drivers proportionally.
   Darkstores (31%) and OSA (25%) together = 56% of the loss.
   Both operate with a 1-day lag — so yesterday's store count drove today's drop.

```
GEOGRAPHIC STORY:
  Others:    Rs.-127,280  OSA=64.6%
  Delhi-NCR: Rs. -65,334  OSA=65.2%
  Bangalore: Rs. -49,573  OSA=69.7%
```
→ "Others" (Tier2/3 cities) accounts for the largest single loss.
   Delhi-NCR and Bangalore are metros with near-critical OSA levels.
   Pattern is SPREAD — not concentrated in 1-2 cities.

```
TOP LOSING SKUs:
  Jaggery Powder 500g:  Rs.-41,140  (-28.4%)
  Sonamasuri Rice 5kg:  Rs.-30,407  (-27.6%)
  Moong Dal 500g:       Rs.-21,940  (-21.2%)
```
→ These specific SKUs dropped 21-33%.
   If market is the cause, all SKUs should fall proportionally.
   Jaggery Powder falling -28.4% vs category -22.1% = slightly worse than market.
   Worth investigating OSA specifically for Jaggery Powder.

```
LEADING ALERTS (forward-looking):
  [HIGH] Darkstore Count declined 2 days (delta=-101)
  [HIGH] Mumbai OSA: 63.7% (below 65% critical)
  [HIGH] Lucknow OSA: 63.4% (critical breach)
```
→ These are predictions for tomorrow (Apr 21).
   Mumbai and Lucknow OSA are already in or near critical territory.
   If not fixed by tomorrow morning, expect further sales drop in those cities.

---

## Recommended Morning Routine

```
1. Run the engine on fresh data (update file paths if new data arrives)
   python advanced_rca_engine.py
   → Takes ~55-95 seconds to complete all phases

2. Check console Quick Summary:
   - Verdict: MARKET_ISSUE or OWN_ISSUE?
   - Leading alerts: how many HIGH alerts?
   - Top driver: what is causing the most loss?

3. If OWN_ISSUE:
   - Go to city_detail in JSON or L7 in Executive Summary
   - Find concentrated cities (CONCENTRATED pattern)
   - Alert ops team for those specific cities

4. If leading alerts are HIGH (especially OSA below 65%):
   - Forward L8 alerts to ops team immediately
   - OSA below 65% = emergency restocking required

5. For deeper investigation:
   - Open llm_evidence_pack.xlsx → filter locality_intelligence by city
   - Look at High_Value_Store_Count_DoD — did we lose quality stores?
   - Paste filtered rows into Claude for SKU-level analysis

6. Paste Executive Summary into Claude for written summary:
   - "Give me a 3-bullet briefing for my manager."
   - "Which city should ops prioritize fixing first?"
```

---

## What the Engine Cannot Do (Limitations)

| Limitation | Why | Workaround |
|------------|-----|------------|
| Confidence is 40% | Only 20 days of data | Add more historical data when available |
| Ad SOV lag not significant | Need 60+ days for Ad SOV significance | Treat as directional, not definitive |
| Network_Strength lag not significant | Same data volume issue | Use HHI and locality score trends instead |
| "Others" Darkstores = 0 | City name mismatch between File 1 and File 2 | Cross-check File 1 Tier2/3 city OSA directly |
| No external signals | No weather/holiday calendar | Manually tag holidays in feedback_overrides.json |
| JSON Evidence Pack is 16MB | Large file | For Claude: use individual Excel sheets, not the full JSON |
| HHI/P75 slow to compute | 16,500 groupby operations with custom lambdas | Runs once per engine run, ~15-25 sec cost |

---

## Troubleshooting

**Error: FileNotFoundError**
```
FileNotFoundError: [Errno 2] No such file or directory: 'D:\2. Area\...'
```
Fix: Check that both CSV files exist at the exact paths in the CONFIG section.

**Error: ModuleNotFoundError**
```
ModuleNotFoundError: No module named 'sklearn'
```
Fix: `pip install scikit-learn`

**Error: UnicodeEncodeError**
Already fixed in the current version (all Rs. symbols are ASCII).
If it recurs, check if someone edited the file and re-introduced special characters.

**Output folder not created**
Check that the `output` directory path is writable.
Current path: `c:\Users\cpsge\.gemini\antigravity\scratch\Project - Sales RCA Reasoning\output\`

**Comparison shows "No data for date"**
The requested date is not in the data files.
Check available dates: Apr 1 – Apr 20, 2026.

---

## Feedback Override System

If the engine attributes a drop to the wrong cause (e.g., a warehouse fire caused
the Apr20 drop, not OSA), create a file named `feedback_overrides.json` in the
project folder:

```json
[
  {
    "date": "2026-04-20",
    "driver": "OSA",
    "actual_cause": "Mumbai DC warehouse fire — force majeure",
    "exclude_from_training": true,
    "added_by": "your_name",
    "added_on": "2026-04-21"
  }
]
```

The engine will read this file (in a future version) and exclude those dates from
baseline and correlation calculations, improving future RCA accuracy.

---

## Output File Naming

Each run creates a folder:
```
output\RCA_[Month] [DD]-[YYYY] [HH]-[MM] [AM/PM]\
Example: output\RCA_April 22-2026 12-09 PM\
```

This means you can run the engine multiple times with different date comparisons
and all outputs are preserved — no overwriting.

Each folder contains:
```
advanced_rca_report.json            (≈54 KB)
advanced_rca_executive_summary.md   (≈14 KB)
llm_evidence_pack.xlsx              (≈2.8 MB)
llm_evidence_pack.json              (≈16 MB)
```
