"""
Advanced Sales RCA Engine -- 24 Mantra Organic (Blinkit)
10-Layer architecture. Outputs timestamped folder + LLM brief.
Run: python advanced_rca_engine.py
"""
import json, os, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
FILE1  = r"D:\2. Area\24 Mantra\April-26\Sales RCA\blinkit-availability-data-april26-day wise.csv"
FILE2  = r"D:\2. Area\24 Mantra\April-26\Sales RCA\blinkit-rca-download-April-26-Daily-City-Comp.csv"
OUTDIR = r"c:\Users\cpsge\.gemini\antigravity\scratch\Project - Sales RCA Reasoning\output"

BRAND_RAW    = "24 mantra organic"        # as stored in data (lowercase)
BRAND_DISPLAY= "24 Mantra Organic"        # for display / reports

COMPARE_DATE1 = "2026-04-19"
COMPARE_DATE2 = "2026-04-20"

# Period comparison (week-level or any date range)
COMPARE_PERIOD1_START = "2026-04-14"   # recent period (e.g. last week)
COMPARE_PERIOD1_END   = "2026-04-20"
COMPARE_PERIOD2_START = "2026-04-07"   # prior period (e.g. week before)
COMPARE_PERIOD2_END   = "2026-04-13"

THR = {
    "OSA_critical":      65.0,
    "OSA_warning":       75.0,
    "Discount_dimret":   20.0,
    "SOV_saturation":    40.0,
    "Dark_metro_min":    30,
    "Pullforward_ratio": 0.5,
    "Anomaly_z":         2.0,
    "Trend_days":        2,
}
METRO = {"mumbai","delhi-ncr","bangalore","chennai","hyderabad","kolkata","pune","ahmedabad"}

# ── OUTPUT FOLDER ─────────────────────────────────────────────────────────────
def make_output_dir() -> Path:
    ts  = datetime.now().strftime("%B %d-%Y %I-%M %p")  # e.g. April 22-2026 12-01 AM
    out = Path(OUTDIR) / f"RCA_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data():
    print("[LOAD] Reading File 1 (availability) ...")
    c1 = ["Date","Store ID","Product ID","Brand","Item ID","Title","Grammage",
          "BGR","Locality City","Locality","Listing %","Avg. OSA %","MRP","SP",
          "Wt. Disc %","SKU Stock Levels","Locality Sales Contribution","Product_type"]
    df1 = pd.read_csv(FILE1, usecols=c1, parse_dates=["Date"], low_memory=False)
    df1["Brand"] = df1["Brand"].str.lower().str.strip()
    df1 = df1[df1["Brand"] == BRAND_RAW].copy()
    df1["Locality City"] = df1["Locality City"].str.lower().str.strip()

    print("[LOAD] Reading File 2 (RCA metrics) ...")
    c2 = ["Date","Product ID","Item ID","Product Name","Grammage","Category","City","Brand",
          "Offtake MRP","Offtake SP","SP","MRP","Units","Wt. OSA %","Wt. Discount %",
          "Category Share","Est. Category Share SP","Overall SOV","Organic SOV","Ad SOV",
          "Wt. PPU (x100)","Product_type"]
    df_all = pd.read_csv(FILE2, usecols=c2, parse_dates=["Date"], low_memory=False)
    df_all["Brand"] = df_all["Brand"].str.lower().str.strip()

    active_stores = df1[df1["Avg. OSA %"] == 100.0].copy()
    active_stores["Is_High_Value"] = (active_stores["Locality Sales Contribution"] >= 0.00168).astype(int)
    
    # Fix City mapping for Delhi-NCR
    ncr_cities = ["delhi", "new delhi", "noida", "gurgaon", "gurugram", "faridabad", "ghaziabad"]
    active_stores["city_key"] = active_stores["Locality City"].str.lower().str.strip()
    active_stores["city_key"] = active_stores["city_key"].apply(lambda c: "delhi-ncr" if c in ncr_cities else c)
    
    dark = active_stores.groupby(["Date","city_key","Product ID"]).agg(
        Darkstore_Count=("Store ID", "size"),
        Network_Strength=("Locality Sales Contribution", "sum"),
        High_Value_Darkstores=("Is_High_Value", "sum")
    ).reset_index()
    dark["Network_Strength"] = dark["Network_Strength"] * 100

    df_all["city_key"] = df_all["City"].str.lower().str.strip()
    df_all = df_all.merge(dark, left_on=["Date","city_key","Product ID"],
                          right_on=["Date","city_key","Product ID"], how="left")
    df_all["Darkstore_Count"] = df_all["Darkstore_Count"].fillna(0)
    df_all["Network_Strength"] = df_all["Network_Strength"].fillna(0)
    df_all["High_Value_Darkstores"] = df_all["High_Value_Darkstores"].fillna(0)

    df24 = df_all[df_all["Brand"] == BRAND_RAW].copy()
    print(f"    File1 rows: {len(df1):,}  |  File2 (24M): {len(df24):,}  |  All brands: {len(df_all):,}")
    return df1, df24, df_all, active_stores


# ── NATIONAL TIME-SERIES ──────────────────────────────────────────────────────
def build_ts(df24: pd.DataFrame) -> pd.DataFrame:
    ts = (df24.groupby("Date").agg(
            Revenue    =("Offtake MRP","sum"),
            Units      =("Units","sum"),
            OSA        =("Wt. OSA %","mean"),
            Discount   =("Wt. Discount %","mean"),
            Ad_SOV     =("Ad SOV","mean"),
            Organic_SOV=("Organic SOV","mean"),
            Overall_SOV=("Overall SOV","mean"),
            Cat_Share  =("Category Share","mean"),
            Darkstores =("Darkstore_Count","sum"),
            Network_Strength =("Network_Strength","sum"),
            High_Value_Darkstores =("High_Value_Darkstores","sum"),
          ).reset_index().sort_values("Date"))
    ts["Day_Num"] = (ts["Date"] - ts["Date"].min()).dt.days
    ts["DOW"]     = ts["Date"].dt.dayofweek
    return ts


# ── LAYER 0: RAW DAILY SNAPSHOT (evidence-first) ─────────────────────────────
def L0_raw_snapshot(ts: pd.DataFrame, n: int = 7) -> dict:
    """
    Raw daily metrics for the last N days.
    This is the evidence base — every insight in L1-L10 must be
    traceable back to these numbers.
    """
    print(f"[L0] Raw daily snapshot (last {n} days) ...")
    last_n = ts.tail(n).copy()
    rows   = []
    for _, r in last_n.iterrows():
        rows.append({
            "date":       r["Date"].strftime("%b %d"),
            "date_full":  r["Date"].strftime("%Y-%m-%d"),
            "dow":        r["Date"].strftime("%a"),
            "revenue_rs": round(float(r["Revenue"]), 0),
            "revenue_L":  f"{float(r['Revenue'])/100000:.1f}L",
            "OSA":        round(float(r["OSA"]),       2) if not pd.isna(r["OSA"])       else None,
            "Darkstores": int(r["Darkstores"])              if not pd.isna(r["Darkstores"]) else None,
            "Ad_SOV":     round(float(r["Ad_SOV"]),    2) if not pd.isna(r["Ad_SOV"])    else None,
            "Discount":   round(float(r["Discount"]),  2) if not pd.isna(r["Discount"])  else None,
        })
    # Print a readable table to console
    hdr = f"{'Date':<10} {'DOW':<4} {'Revenue':>10}  {'OSA':>7}  {'Darkstores':>11}  {'Ad SOV':>7}  {'Discount':>9}"
    print("    " + hdr)
    print("    " + "-"*len(hdr))
    for r in rows:
        rev_str = f"Rs.{r['revenue_rs']:,.0f} ({r['revenue_L']})"
        print(f"    {r['date']:<10} {r['dow']:<4} {rev_str:>22}  {str(r['OSA']):>7}  {str(r['Darkstores']):>11}  {str(r['Ad_SOV']):>7}  {str(r['Discount']):>9}")

    # DoD changes for last row vs second-last
    dods = {}
    if len(rows) >= 2:
        d2, d1 = rows[-1], rows[-2]
        for col in ["revenue_rs","OSA","Darkstores","Ad_SOV","Discount"]:
            if d1[col] is not None and d2[col] is not None and d1[col] != 0:
                dods[col] = round((d2[col]-d1[col])/abs(d1[col])*100, 1)

    return {"last_n_days": rows, "dod_changes": dods, "n": n}



def L1_baseline(ts: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    print("[L1] Baseline decomposition ...")
    if len(ts) < 7:
        return {"note": "Need >=7 days"}, ts

    y  = ts["Revenue"].values.astype(float)
    x  = ts["Day_Num"].values.reshape(-1,1)
    lr = LinearRegression().fit(x, y)
    trend     = lr.predict(x)
    detrended = y - trend
    dow       = ts["DOW"].values
    dow_fx    = np.array([detrended[dow==d].mean() if (dow==d).any() else 0 for d in dow])
    residual  = detrended - dow_fx

    ts = ts.copy()
    ts["Trend"]    = trend
    ts["DOW_FX"]   = dow_fx
    ts["Residual"] = residual
    ts["Resid_Z"]  = stats.zscore(residual) if residual.std() > 0 else np.zeros(len(residual))

    anom = ts[ts["Resid_Z"].abs() > THR["Anomaly_z"]]
    result = {
        "trend_slope_day":    round(float(lr.coef_[0]), 2),
        "trend_r2":           round(float(lr.score(x,y)), 4),
        "dow_effects":        {str(d): round(float(detrended[dow==d].mean()),2)
                               for d in range(7) if (dow==d).any()},
        "true_anomaly_dates": anom["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "anomaly_details":    anom[["Date","Revenue","Resid_Z"]].assign(
                                Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"),
                                Revenue=lambda d: d["Revenue"].round(0),
                                Resid_Z=lambda d: d["Resid_Z"].round(2)
                              ).to_dict(orient="records"),
    }
    print(f"    Slope: Rs.{result['trend_slope_day']}/day | Anomalies: {len(anom)}")
    return result, ts


# ── LAYER 4: LAG ANALYSIS ─────────────────────────────────────────────────────
def L4_lag(ts: pd.DataFrame) -> dict:
    print("[L4] Lag correlation ...")
    result = {}
    for col in ["OSA","Discount","Ad_SOV","Network_Strength"]:
        best_lag, best_r, best_p = 0, 0.0, 1.0
        table = []
        for lag in range(4):
            sh   = ts[col].shift(lag)
            mask = sh.notna() & ts["Revenue"].notna()
            if mask.sum() < 5: continue
            r, p = stats.pearsonr(sh[mask], ts["Revenue"][mask])
            table.append({"lag_days": lag, "r": round(float(r),4), "p": round(float(p),4)})
            if abs(r) > abs(best_r): best_lag, best_r, best_p = lag, r, p

        # Build explicit lag validation examples (last 3 pairs)
        lag_examples = []
        for i in range(max(0, len(ts)-3-best_lag), len(ts)-best_lag):
            if i < 0 or i+best_lag >= len(ts): continue
            driver_date = ts.iloc[i]["Date"].strftime("%b %d")
            rev_date    = ts.iloc[i+best_lag]["Date"].strftime("%b %d")
            driver_val  = ts.iloc[i][col]
            rev_val     = ts.iloc[i+best_lag]["Revenue"]
            if pd.isna(driver_val) or pd.isna(rev_val): continue
            lag_examples.append(
                f"{driver_date} {col}={driver_val:.2f} -> {rev_date} Revenue=Rs.{rev_val:,.0f}"
            )

        result[col] = {
            "best_lag_days":   best_lag,
            "best_r":          round(float(best_r),4),
            "best_p":          round(float(best_p),4),
            "significant":     best_p < 0.05,
            "direction":       "positive" if best_r > 0 else "negative",
            "lag_table":       table,
            "lag_validation":  lag_examples,
        }
        print(f"    {col}: lag={best_lag}d  r={best_r:.3f}  p={best_p:.3f}")
        for ex in lag_examples:
            print(f"        {ex}")
    return result


# ── LAYER 2: DRIVER INTERACTIONS ──────────────────────────────────────────────
def L2_interactions(ts):
    print("[L2] Driver interactions ...")
    latest = ts.iloc[-1]
    osa  = float(latest["OSA"])        if not pd.isna(latest["OSA"])        else 0
    disc = float(latest["Discount"])   if not pd.isna(latest["Discount"])   else 0
    sov  = float(latest["Ad_SOV"])     if not pd.isna(latest["Ad_SOV"])     else 0
    dark = float(latest["Network_Strength"]) if not pd.isna(latest["Network_Strength"]) else 0
    checks = [
        ("SOV x OSA wastage",          sov>5 and osa<THR["OSA_warning"],  "HIGH",
         f"Ad SOV {sov:.1f}% active but OSA only {osa:.1f}% -- product unavailable. SOV wasted. Fix OSA first."),
        ("OSA x Discount double-penalty", osa<THR["OSA_critical"] and disc<5, "HIGH",
         f"Critical OSA {osa:.1f}% AND low Discount {disc:.1f}% -- compounding drag, both need urgent fix."),
        ("Discount x Net Strength",      disc>10 and dark<80, "MEDIUM",
         f"Discount {disc:.1f}% running but Network Strength is only {dark:.0f}% -- promo not reaching potential."),
        ("SOV x OSA amplified",        sov>10 and osa>THR["OSA_warning"],  "POSITIVE",
         f"High SOV {sov:.1f}% + healthy OSA {osa:.1f}% -- both levers aligned, amplified impact."),
    ]
    active    = [c for c in checks if c[1]]
    warnings  = [c[3] for c in active if c[2] in ("HIGH","MEDIUM")]
    positives = [c[3] for c in active if c[2]=="POSITIVE"]
    print(f"    Active: {len(active)}  Warnings: {len(warnings)}")
    return {"interactions":[{"pair":c[0],"severity":c[2],"insight":c[3]} for c in active],
            "warnings":warnings,"positives":positives}


# ── LAYER 3: THRESHOLD LOGIC ──────────────────────────────────────────────────
def L3_thresholds(ts, lag):
    print("[L3] Thresholds + zone context ...")
    latest = ts.iloc[-1]
    # Zone definitions per driver
    zone_defs = {
        "OSA":      [(75,  100, "SAFE",     ">75%   -> safe zone"),
                     (65,   75, "RISK",     "65-75% -> risk zone (drag starts)"),
                     (0,    65, "CRITICAL", "<65%   -> critical zone (collapse risk)")],
        "Discount": [(0,   16,  "SAFE",     "<16%   -> safe zone"),
                     (16,  20,  "RISK",     "16-20% -> risk zone (dimret starting)"),
                     (20, 100,  "CRITICAL", ">20%   -> critical zone (base sales erode)")],
        "Ad_SOV":   [(0,   28,  "SAFE",     "<28%   -> normal spend"),
                     (28,  40,  "RISK",     "28-40% -> diminishing returns"),
                     (40, 100,  "CRITICAL", ">40%   -> saturation, cut spend")],
    }
    cfg = {
        "OSA":      ("OSA",     THR["OSA_critical"],   THR["OSA_warning"],       "below_bad"),
        "Discount": ("Discount",THR["Discount_dimret"], THR["Discount_dimret"]*0.8,"above_bad"),
        "Ad_SOV":   ("Ad_SOV",  THR["SOV_saturation"], THR["SOV_saturation"]*0.7, "above_dimret"),
    }
    result = {}
    for name,(col,crit,warn,dir_) in cfg.items():
        val       = float(latest[col]) if not pd.isna(latest[col]) else None
        below_avg = ts[ts[col]< crit]["Revenue"].mean()
        above_avg = ts[ts[col]>=crit]["Revenue"].mean()
        pen = (round((float(above_avg-below_avg)/float(above_avg)*100),1)
               if (above_avg and not np.isnan(above_avg) and above_avg>0) else None)
        status = ("CRITICAL" if (dir_=="below_bad" and val is not None and val<crit) else
                  "WARNING"  if (dir_=="below_bad" and val is not None and val<warn) else
                  "CRITICAL" if (dir_=="above_bad" and val is not None and val>crit) else
                  "WARNING"  if (dir_=="above_bad" and val is not None and val>warn) else
                  "MONITOR"  if dir_=="above_dimret" else "OK")
        # Determine which zone current value sits in
        current_zone = "UNKNOWN"
        for lo, hi, zname, zdesc in zone_defs.get(name, []):
            if val is not None and lo <= val < hi:
                current_zone = zname
                break
        # Trend: last 3 days of this driver
        trend_vals = []
        for _, row in ts.tail(3).iterrows():
            v = row[col]
            trend_vals.append({
                "date": row["Date"].strftime("%b %d"),
                "value": round(float(v), 2) if not pd.isna(v) else None
            })
        result[name] = {
            "current":         val,
            "critical":        crit,
            "warning":         warn,
            "status":          status,
            "current_zone":    current_zone,
            "zone_bands":      zone_defs.get(name, []),
            "sales_penalty_pct": pen,
            "best_lag_days":   lag.get(col, lag.get(name,{})).get("best_lag_days",0),
            "last_3_days":     trend_vals,
        }
        zone_str = f"[{current_zone}]" if current_zone else ""
        print(f"    {name}: {val:.2f}% -> {status} {zone_str}")
        for t in trend_vals:
            print(f"        {t['date']}: {t['value']}")
    return result


# ── LAYER 5: PULLFORWARD ──────────────────────────────────────────────────────
def L5_pullforward(ts):
    print("[L5] Pullforward filter ...")
    pf, real = [], []
    if len(ts)<3: return {"pullforward_events":pf,"real_drops":real}
    y,disc,dt = ts["Revenue"].values,ts["Discount"].values,ts["Date"].dt.strftime("%Y-%m-%d").values
    for i in range(1,len(y)-1):
        if disc[i]-disc[i-1]>5 and y[i]>y[i-1]:
            spike = (y[i]-y[i-1])/y[i-1] if y[i-1]>0 else 0
            drop  = (y[i]-y[i+1])/y[i]   if y[i]>0   else 0
            if drop>0:
                ev={"spike_date":dt[i],"drop_date":dt[i+1],
                    "spike_pct":round(spike*100,1),"drop_pct":round(drop*100,1)}
                (pf if drop<=spike*THR["Pullforward_ratio"] else real).append(ev)
    print(f"    Pullforward: {len(pf)}  Real: {len(real)}")
    return {"pullforward_events":pf,"real_drops":real}


# -- LAYER 6: COMPETITIVE & MARKET SEPARATION ---------------------------------
def L6_competitive(df24, df_all):
    """
    Category Share column = 24M's % share within a specific sub-category
    (e.g. 6.46 means 24M holds 6.46% of the Jaggery category in that city).
    It is already in PERCENTAGE format (0-100 scale).

    Correct category revenue derivation:
        Implied_Cat_Rev = Offtake_MRP / (Category_Share / 100)

    To avoid double-counting (multiple 24M SKUs share the same category pool),
    we group by (Date, City, Category) and take mean of implied values per group,
    then sum groups to get total category revenue per day.
    """
    print("[L6] Competitive separation (correct category back-calc) ...")

    valid = df24[
        df24["Offtake MRP"].notna() &
        df24["Category Share"].notna() &
        (df24["Category Share"] > 0)
    ].copy()
    # Category Share is already %, e.g. 6.46 = 6.46%
    valid["Implied_Cat_Rev"] = valid["Offtake MRP"] / (valid["Category Share"] / 100)

    # Per (Date, City, Category): mean implied category revenue to avoid double-count
    cat_grp = valid.groupby(["Date", "City", "Category"]).agg(
        Brand_Rev_in_grp=("Offtake MRP", "sum"),
        Cat_Rev_in_grp=("Implied_Cat_Rev", "mean"),
    ).reset_index()

    # Daily national totals
    daily = cat_grp.groupby("Date").agg(
        Brand_Rev=("Brand_Rev_in_grp", "sum"),
        Total_Cat_Rev=("Cat_Rev_in_grp", "sum"),
    ).reset_index().sort_values("Date")

    daily["Brand_Share_Pct"] = daily["Brand_Rev"] / daily["Total_Cat_Rev"] * 100
    daily["Cat_DoD"]         = daily["Total_Cat_Rev"].pct_change() * 100
    daily["Brand_DoD"]       = daily["Brand_Rev"].pct_change() * 100
    daily["Share_DoD"]       = daily["Brand_Share_Pct"].diff()
    daily["Date_str"]        = daily["Date"].dt.strftime("%Y-%m-%d")

    last      = daily.iloc[-1]
    bdod      = float(last["Brand_DoD"])       if not pd.isna(last["Brand_DoD"])       else 0
    cdod      = float(last["Cat_DoD"])         if not pd.isna(last["Cat_DoD"])         else 0
    share     = float(last["Brand_Share_Pct"]) if not pd.isna(last["Brand_Share_Pct"]) else 0
    share_chg = float(last["Share_DoD"])       if not pd.isna(last["Share_DoD"])       else 0

    # Verdict: category is the true market benchmark
    if bdod < 0 and cdod < 0 and abs(cdod) >= abs(bdod) * 0.7:
        verdict = "MARKET_ISSUE"
        expl    = (f"Both brand ({bdod:.1f}%) and category ({cdod:.1f}%) declined. "
                   f"Market-level headwind, not specific to {BRAND_DISPLAY}.")
    elif bdod < 0 and cdod > 0:
        verdict = "OWN_ISSUE"
        expl    = (f"Brand fell {bdod:.1f}% while category GREW {cdod:.1f}%. "
                   f"Market was healthy -- this is entirely a {BRAND_DISPLAY} own-factor problem. "
                   f"Share dropped {abs(share_chg):.2f} pts to {share:.2f}%.")
    elif bdod < 0 and abs(cdod) < 2:
        verdict = "OWN_ISSUE"
        expl    = (f"Brand fell {bdod:.1f}% but category was flat ({cdod:.1f}%). "
                   f"Own levers are the cause. Share dropped {abs(share_chg):.2f} pts to {share:.2f}%.")
    elif bdod < cdod:
        verdict = "OWN_ISSUE"
        expl    = (f"Brand {bdod:.1f}% vs category {cdod:.1f}% -- "
                   f"underperforming market. Share: {share:.2f}% ({share_chg:+.2f} pts).")
    else:
        verdict = "OUTPERFORMING"
        expl    = (f"Brand {bdod:.1f}% vs category {cdod:.1f}% -- beating market. "
                   f"Share: {share:.2f}% ({share_chg:+.2f} pts).")

    print(f"    Verdict: {verdict}  Brand: {bdod:.1f}%  Category: {cdod:.1f}%  Share: {share:.2f}%")
    return {
        "daily": daily[["Date_str","Brand_Rev","Total_Cat_Rev","Brand_Share_Pct",
                         "Cat_DoD","Brand_DoD","Share_DoD"]].round(2).to_dict(orient="records"),
        "latest_verdict":     verdict,
        "latest_explanation": expl,
        "brand_dod":          round(bdod, 2),
        "cat_dod":            round(cdod, 2),
        "brand_share_pct":    round(share, 2),
        "share_point_change": round(share_chg, 2),
        "note": ("Category Share in data = 24M's % within each sub-category (Jaggery, Rice, etc). "
                 "Total category revenue back-calculated per (Date,City,Category) group to avoid double-count."),
    }

# ── LAYER 7: GEOGRAPHIC CASCADE ───────────────────────────────────────────────
def L7_geographic(df24, l6_verdict=""):
    print("[L7] Geographic cascade ...")
    city = (df24.groupby(["Date","City"]).agg(
                Revenue=("Offtake MRP","sum"),OSA=("Wt. OSA %","mean"),
                Discount=("Wt. Discount %","mean"),Ad_SOV=("Ad SOV","mean"),
                Network_Strength=("Network_Strength","sum"))
            .reset_index().sort_values(["City","Date"]))
    city["Tier"] = city["City"].str.lower().str.strip().apply(lambda c:"Metro" if c in METRO else "Tier2/3")
    dates = sorted(city["Date"].unique())
    d2,d1 = dates[-1],(dates[-2] if len(dates)>1 else dates[-1])
    last = city[city["Date"]==d2].copy()
    prev = city[city["Date"]==d1][["City","Revenue"]].rename(columns={"Revenue":"Prev"})
    last = last.merge(prev,on="City",how="left")
    last["Change"] = last["Revenue"]-last["Prev"]
    total_chg = float(last["Change"].sum())
    keep = ["City","Tier","Revenue","Change","OSA","Ad_SOV","Network_Strength"]
    losers  = last.nsmallest(5,"Change")[keep].round(1).to_dict(orient="records")
    gainers = last.nlargest(5, "Change")[keep].round(1).to_dict(orient="records")
    losing  = last[last["Change"]<0]
    top2_share = (float(losing.nsmallest(2,"Change")["Change"].sum()/total_chg)
                  if total_chg<0 and len(losing)>=2 else 0.0)
    geo_pattern = "CONCENTRATED" if top2_share>0.7 else "SPREAD"

    # OVERRIDE: if L6 already confirmed OWN_ISSUE (category grew, brand fell),
    # geo SPREAD does NOT mean macro — it means the own-factor is national in scope
    if l6_verdict == "OWN_ISSUE" and geo_pattern == "SPREAD":
        effective_pattern = "SPREAD_OWN"
        note = ("Drop spread nationally, BUT category grew on same day (L6=OWN_ISSUE). "
                "SPREAD pattern here means the own-factor (OSA, darkstores, discount) "
                "degraded across all cities simultaneously -- NOT a macro market issue.")
    elif geo_pattern == "CONCENTRATED":
        effective_pattern = "CONCENTRATED"
        note = "Top 2 cities >70% of drop -- likely OPERATIONAL issue in those cities (OSA/darkstore outage)."
    else:
        effective_pattern = "SPREAD"
        note = "Drop spread across cities -- consistent with macro/market issue."

    print(f"    Geo pattern: {geo_pattern} (effective: {effective_pattern})  L6 override: {l6_verdict}")
    return {"top_losing_cities":losers,"top_gaining_cities":gainers,
            "drop_pattern":effective_pattern,"raw_geo_pattern":geo_pattern,
            "drop_pattern_note":note,"total_national_change":round(total_chg,0)}


# ── LAYER 8: LEADING INDICATORS ──────────────────────────────────────────────
def L8_leading(ts, df24):
    print("[L8] Leading indicators (with daily values) ...")
    alerts, N = [], THR["Trend_days"]

    def chk(series, label, col_name, direction, thr_delta, dates_series):
        vals  = series.dropna().values
        dates = dates_series[series.notna()].values
        if len(vals) < N+1: return
        recent_vals  = vals[-N:]
        recent_dates = dates[-N:]
        # Build daily trend string
        trend_str = "  |  ".join(
            f"{pd.Timestamp(d).strftime('%b %d')} = {v:.2f}"
            for d, v in zip(recent_dates, recent_vals)
        )
        net_chg = float(recent_vals[-1] - recent_vals[0])
        if direction == "down" and all(recent_vals[i] < recent_vals[i-1] for i in range(1, N)):
            alerts.append({
                "indicator":  label,
                "alert":      f"{label} declined {N} consecutive days",
                "daily_trend":trend_str,
                "net_change": round(net_chg, 2),
                "severity":   "HIGH" if abs(net_chg) > thr_delta else "MEDIUM",
                "prediction": "Sales at risk in next 2-3 days if trend continues.",
                "current":    round(float(recent_vals[-1]), 2),
            })
        elif direction == "up" and all(recent_vals[i] > recent_vals[i-1] for i in range(1, N)):
            alerts.append({
                "indicator":  label,
                "alert":      f"{label} rising {N} days straight (pullforward risk)",
                "daily_trend":trend_str,
                "net_change": round(net_chg, 2),
                "severity":   "MEDIUM",
                "prediction": "Watch for demand pullforward correction.",
                "current":    round(float(recent_vals[-1]), 2),
            })

    chk(ts["OSA"],              "National OSA (%)",   "OSA",              "down",  5, ts["Date"])
    chk(ts["Ad_SOV"],           "Ad SOV (%)",         "Ad_SOV",           "down",  2, ts["Date"])
    chk(ts["Network_Strength"], "Network Strength",   "Network_Strength", "down",  2, ts["Date"])
    chk(ts["Discount"],         "Discount (%)",       "Discount",         "up",    5, ts["Date"])

    city_osa = (df24.groupby(["Date","City"])["Wt. OSA %"].mean()
                .reset_index().sort_values(["City","Date"]))
    for city_name, grp in city_osa.groupby("City"):
        vals  = grp["Wt. OSA %"].values
        dates = grp["Date"].values
        if len(vals) >= N+1:
            recent_v = vals[-N:]
            recent_d = dates[-N:]
            if all(recent_v[i] < recent_v[i-1] for i in range(1, N)):
                trend_str = "  |  ".join(
                    f"{pd.Timestamp(d).strftime('%b %d')} = {v:.1f}%"
                    for d, v in zip(recent_d, recent_v)
                )
                alerts.append({
                    "indicator":   f"OSA:{city_name}",
                    "alert":       f"{city_name} OSA declining {N} days straight",
                    "daily_trend": trend_str,
                    "net_change":  round(float(recent_v[-1]-recent_v[0]), 2),
                    "severity":    "HIGH" if recent_v[-1] < THR["OSA_warning"] else "MEDIUM",
                    "prediction":  f"Sales at risk in {city_name} within 2-3 days.",
                    "current":     round(float(recent_v[-1]), 2),
                })
    print(f"    Alerts raised: {len(alerts)}")
    for a in alerts[:5]:  # show first 5 in console
        print(f"    [{a['severity']}] {a['alert']}")
        print(f"        Trend: {a['daily_trend']}  (net {a['net_change']:+.2f})")
    return {"alerts": alerts, "total_alerts": len(alerts)}

# ── HISTORICAL EVIDENCE ───────────────────────────────────────────────────────
def find_historical_evidence(ts, driver_col, target_delta, expected_impact, current_d1, current_d2):
    if len(ts) < 5 or target_delta == 0: return None
    
    sign = 1 if target_delta > 0 else -1
    min_mag = abs(target_delta) * 0.7
    max_mag = abs(target_delta) * 1.3
    
    best_match, min_diff = None, float('inf')
    
    for i in range(1, len(ts)):
        d1 = ts.iloc[i-1]["Date"]
        d2 = ts.iloc[i]["Date"]
        
        if d1 >= pd.Timestamp(current_d1) or d2 >= pd.Timestamp(current_d1):
            continue
            
        v1 = ts.iloc[i-1][driver_col]
        v2 = ts.iloc[i][driver_col]
        if pd.isna(v1) or pd.isna(v2): continue
        
        delta = v2 - v1
        if delta * sign > 0 and min_mag <= abs(delta) <= max_mag:
            rev1, rev2 = ts.iloc[i-1]["Revenue"], ts.iloc[i]["Revenue"]
            rev_delta = rev2 - rev1
            # Only accept history where revenue moved in the expected direction
            if (expected_impact > 0 and rev_delta > 0) or (expected_impact < 0 and rev_delta < 0):
                diff = abs(abs(delta) - abs(target_delta))
                if diff < min_diff:
                    min_diff = diff
                    best_match = {"hist_d1": d1.strftime("%b %d"), "hist_d2": d2.strftime("%b %d"),
                                  "driver_delta": delta, "rev_delta": rev_delta}
                
    if best_match:
        dir_w = "dropped" if target_delta < 0 else "rose"
        r_dir = "dropped" if best_match["rev_delta"] < 0 else "rose"
        return (f"Historical Proof: {best_match['hist_d1']}->{best_match['hist_d2']}, {driver_col} {dir_w} "
                f"by {abs(best_match['driver_delta']):.1f} causing Revenue to {r_dir} by Rs.{abs(best_match['rev_delta']):,.0f}.")
    return "Historical Proof: No direct past match found validating this specific impact."

# ── LAYER 9: FINANCIAL ATTRIBUTION ───────────────────────────────────────────
def L9_attribution(ts, lag):
    print("[L9] Financial attribution ...")
    if len(ts)<3: return {"note":"Insufficient data"}
    latest,prev = ts.iloc[-1],ts.iloc[-2]
    total_delta = float(latest["Revenue"]-prev["Revenue"])
    drivers = ["OSA","Discount","Ad_SOV","Network_Strength"]
    sens = _fit_sku_city_regression(ts, drivers)
    attr = []
    for d in drivers:
        curr = float(latest[d]) if not pd.isna(latest[d]) else 0
        base = float(prev[d])   if not pd.isna(prev[d])   else 0
        dd   = curr-base
        coef = sens[d]["coef"]
        impact = dd * coef
        
        if impact != 0:
            hist_ev = (f"Multivariate OLS over 20 days controls for confounders. "
                       f"Every 1 unit change in {d} explains Rs. {coef:,.0f} impact independently. "
                       f"Today's {dd:.2f} delta * {coef:,.0f} = Rs. {impact:,.0f}")
        else:
            hist_ev = "N/A - Impact nullified."
            
        driver_name = d
        custom_narrative = None
        if d == "Network_Strength":
            d_curr = latest["Darkstores"] if "Darkstores" in latest and not pd.isna(latest["Darkstores"]) else 0
            d_base = prev["Darkstores"] if "Darkstores" in prev and not pd.isna(prev["Darkstores"]) else 0
            dark_delta = d_curr - d_base
            net_delta = dd
            store_action = f"lost {abs(dark_delta):.0f}" if dark_delta < 0 else (f"gained {dark_delta:.0f}" if dark_delta > 0 else "saw no change in")
            cap_action = f"wiped out {abs(net_delta):.1f}%" if net_delta < 0 else (f"added {net_delta:.1f}%" if net_delta > 0 else "didn't change")
            imp_action = f"causing a drop of Rs. {abs(impact):,.0f}" if impact < 0 else f"driving a gain of Rs. {abs(impact):,.0f}"
            custom_narrative = f"You {store_action} physical stores, which {cap_action} of your sales capacity ({base:.1f}% -> {curr:.1f}%), {imp_action}."
            driver_name = "Network_Capacity"

        attr.append({"driver":driver_name,"current":round(curr,2),"baseline":round(base,2),
                     "driver_delta":round(dd,2),"revenue_impact_rs":round(impact,0),
                     "share_pct":round(impact/total_delta*100,1) if total_delta!=0 else 0,
                     "weight":round(coef,4),"lag_days":lag.get(d,{}).get("best_lag_days",0),
                     "historical_evidence": hist_ev, "custom_narrative": custom_narrative})
    attr.sort(key=lambda x:abs(x["revenue_impact_rs"]),reverse=True)
    top = attr[0]["driver"] if attr else "Unknown"
    print(f"    Delta: Rs.{total_delta:.0f}  Top: {top}")
    return {"total_delta_rs":round(total_delta,0),
            "analysis_date":latest["Date"].strftime("%Y-%m-%d"),
            "baseline_date":prev["Date"].strftime("%Y-%m-%d"),
            "driver_attributions":attr,"top_driver":top}


# ── LAYER 10: CONFIDENCE & FEEDBACK ──────────────────────────────────────────
def L10_feedback(l1, l9, lag, n_days):
    print("[L10] Confidence scoring ...")
    r2    = l1.get("trend_r2", 0)
    top   = l9.get("top_driver", "OSA")
    top_r = abs(lag.get(top, {}).get("best_r", 0))
    n_an  = len(l1.get("true_anomaly_dates", []))
    score = min(100, int(r2*30 + top_r*40 + min(n_days,30)/30*20 + (10 if n_an>0 else 5)))

    # Build explicit reasons
    reasons, actions = [], []
    if n_days < 30:
        reasons.append(f"Only {n_days} days of data (need 30+ for strong stats)")
        actions.append("Accumulate 30+ days then re-run for higher confidence")
    if r2 < 0.1:
        reasons.append(f"Trend R2={r2:.3f} (very low -- linear trend barely fits data)")
        actions.append("Treat trend slope as directional only, not precise")
    if top_r < 0.5:
        reasons.append(f"Top driver correlation r={top_r:.3f} (moderate -- not definitive)")
        actions.append("Cross-validate with ops team before large interventions")
    for d in ["OSA","Discount","Ad_SOV","Network_Strength"]:
        info = lag.get(d, {})
        if not info.get("significant", False):
            reasons.append(f"{d} lag correlation not statistically significant (p={info.get('best_p',1):.3f})")
    if n_an == 0:
        reasons.append("No true statistical anomalies detected (drops within expected variance)")
        actions.append("Focus on trend direction rather than single-day anomaly investigation")

    interp = ("HIGH -- trust this RCA for decisions" if score >= 70
              else "MEDIUM -- directionally correct, validate 1-2 key claims" if score >= 50
              else "LOW -- use as hypothesis only, validate with raw data before acting")

    guardrail = (
        f"WARNING: Confidence is LOW ({score}%). "
        f"Before acting on this RCA: (1) Check L0 raw table to verify numbers, "
        f"(2) Ask ops team if any external event occurred on these dates, "
        f"(3) Do not increase/cut ad spend based on this alone."
        if score < 50 else
        f"MEDIUM confidence ({score}%). Directional findings are reliable. "
        f"Validate the top driver before large budget moves."
        if score < 70 else
        f"HIGH confidence ({score}%). RCA findings are statistically grounded."
    )

    print(f"    Confidence: {score}%  ({interp.split(' --')[0]})")
    for r in reasons:
        print(f"        Reason: {r}")

    return {
        "confidence_score":   score,
        "interpretation":     interp,
        "guardrail_message":  guardrail,
        "reasons_for_score":  reasons,
        "recommended_actions":actions,
        "factors": {
            "trend_r2":         round(r2, 4),
            "top_driver_r":     round(top_r, 4),
            "days_of_data":     n_days,
            "anomalies_detected": n_an,
        },
        "override_instructions":
            "Add wrong findings to feedback_overrides.json: "
            "{date, driver, actual_cause, exclude_from_training:true}",
    }


def _fit_sku_city_regression_dynamic(history_df):
    """Dynamically scan up to 30 signals, pick best non-collinear ones, run multivariate OLS."""
    import numpy as np
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    import warnings
    
    # Pre-scan univariate
    candidates = [c for c in history_df.columns if c not in ["Date", "Revenue", "DOW"]]
    scored = []
    
    for feat in candidates:
        best_r, best_lag, best_p = 0, 0, 1.0
        for lag in range(4):
            sh = history_df[feat].shift(lag)
            mask = sh.notna() & history_df['Revenue'].notna()
            if mask.sum() < 5 or sh[mask].nunique() <= 1: continue
            try:
                r, p = stats.pearsonr(sh[mask], history_df['Revenue'][mask])
            except: continue
            if not np.isnan(r) and abs(r) > abs(best_r):
                best_r, best_lag, best_p = r, lag, p
        if best_r != 0:
            scored.append({"feat": feat, "r": best_r, "r2": best_r**2, "p": best_p, "lag": best_lag})
            
    scored.sort(key=lambda x: -x["r2"])
    
    # Select best non-redundant (max 7 variables for 20 days of data)
    selected = []
    for s in scored:
        if s["p"] < 0.20:
            # simple collinearity check could go here, for now just pick top
            selected.append(s)
            if len(selected) >= 7: break
            
    if not selected:
        return {}, []
        
    drivers = [s["feat"] for s in selected]
    
    # Build aligned DF
    ols_df = history_df[["Revenue"]].copy()
    for s in selected:
        ols_df[s["feat"]] = history_df[s["feat"]].shift(s["lag"])
    ols_df = ols_df.dropna()
    
    n = len(ols_df)
    p_len = len(drivers)
    sens = {}
    
    if n >= p_len + 2:
        X = ols_df[drivers].values
        y = ols_df["Revenue"].values
        reg = LinearRegression().fit(X, y)
        coefs = reg.coef_
        r2_model = reg.score(X, y)
        for i, s in enumerate(selected):
            feat = s["feat"]
            sens[feat] = {"coef": coefs[i], "p": s["p"], "r2": r2_model, "lag": s["lag"], "r2_uni": s["r2"]}
    else:
        # Fallback to univariate if very short on data
        for s in selected:
            feat = s["feat"]
            sens[feat] = {"coef": 0.0, "p": s["p"], "r2": s["r2"], "lag": s["lag"], "r2_uni": s["r2"]}
            sh = history_df[feat].shift(s["lag"])
            mask = sh.notna() & history_df["Revenue"].notna()
            if mask.sum() >= 5:
                if len(sh[mask].unique()) > 1:
                    res = stats.linregress(sh[mask], history_df["Revenue"][mask])
                    if not np.isnan(res.slope):
                        sens[feat]["coef"] = res.slope
                        
    return sens, selected

# ── SKU DEEP DIVE ─────────────────────────────────────────


def _fit_sku_city_regression_dynamic(history_df):
    """Dynamically scan up to 30 signals, pick best non-collinear ones, run multivariate OLS."""
    import numpy as np
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    import warnings
    
    # Pre-scan univariate
    candidates = [c for c in history_df.columns if c not in ["Date", "Revenue", "DOW"]]
    scored = []
    
    for feat in candidates:
        best_r, best_lag, best_p = 0, 0, 1.0
        for lag in range(4):
            sh = history_df[feat].shift(lag)
            mask = sh.notna() & history_df['Revenue'].notna()
            if mask.sum() < 5 or sh[mask].nunique() <= 1: continue
            try:
                r, p = stats.pearsonr(sh[mask], history_df['Revenue'][mask])
            except: continue
            if not np.isnan(r) and abs(r) > abs(best_r):
                best_r, best_lag, best_p = r, lag, p
        if best_r != 0:
            scored.append({"feat": feat, "r": best_r, "r2": best_r**2, "p": best_p, "lag": best_lag})
            
    scored.sort(key=lambda x: -x["r2"])
    
    # Select best non-redundant (max 7 variables for 20 days of data)
    selected = []
    for s in scored:
        if s["p"] < 0.20:
            selected.append(s)
            if len(selected) >= 7: break
            
    if not selected:
        return {}, []
        
    drivers = [s["feat"] for s in selected]
    
    # Build aligned DF
    ols_df = history_df[["Revenue"]].copy()
    for s in selected:
        ols_df[s["feat"]] = history_df[s["feat"]].shift(s["lag"])
    ols_df = ols_df.dropna()
    
    n = len(ols_df)
    p_len = len(drivers)
    sens = {}
    
    if n >= p_len + 2:
        X = ols_df[drivers].values
        y = ols_df["Revenue"].values
        reg = LinearRegression().fit(X, y)
        coefs = reg.coef_
        r2_model = reg.score(X, y)
        for i, s in enumerate(selected):
            feat = s["feat"]
            sens[feat] = {"coef": coefs[i], "p": s["p"], "r2": r2_model, "lag": s["lag"], "r2_uni": s["r2"]}
    else:
        # Fallback to univariate if very short on data
        for s in selected:
            feat = s["feat"]
            sens[feat] = {"coef": 0.0, "p": s["p"], "r2": s["r2"], "lag": s["lag"], "r2_uni": s["r2"]}
            sh = history_df[feat].shift(s["lag"])
            mask = sh.notna() & history_df["Revenue"].notna()
            if mask.sum() >= 5:
                if len(sh[mask].unique()) > 1:
                    res = stats.linregress(sh[mask], history_df["Revenue"][mask])
                    if not np.isnan(res.slope):
                        sens[feat]["coef"] = res.slope
                        
    return sens, selected


# ── SKU DEEP DIVE ─────────────────────────────────────────

def sku_deep_dive(df24, df_all, df1, sku_id, sku_name, d1_str, d2_str):
    import pandas as pd
    import numpy as np
    
    sku_df = df24[df24["Product ID"] == sku_id].copy()
    if sku_df.empty: return {}
    
    category = sku_df["Category"].iloc[0] if "Category" in sku_df.columns else "Unknown"
    comp = df_all[(df_all["Category"] == category) & (df_all["Brand"] != "24 mantra organic")].copy()
    
    df1_sku = df1[df1["Product ID"] == sku_id].copy()
    
    d1, d2 = pd.Timestamp(d1_str), pd.Timestamp(d2_str)
    
    # 1. Identify top dropping cities for this SKU
    c_rev1 = sku_df[sku_df["Date"] == d1].groupby("City")["Offtake MRP"].sum()
    c_rev2 = sku_df[sku_df["Date"] == d2].groupby("City")["Offtake MRP"].sum()
    c_deltas = c_rev2.sub(c_rev1, fill_value=0).sort_values()
    
    total_rev_d1 = sku_df[sku_df["Date"] == d1]["Offtake MRP"].sum()
    total_rev_d2 = sku_df[sku_df["Date"] == d2]["Offtake MRP"].sum()
    total_drop = total_rev_d2 - total_rev_d1
    
    dropping_cities = c_deltas[c_deltas < -500].index.tolist()
    if not dropping_cities and len(c_deltas) > 0:
        dropping_cities = [c_deltas.index[0]]
        
    city_col_f1 = "Locality City" if "Locality City" in df1_sku.columns else "City" if "City" in df1_sku.columns else None
    
    city_results = []
    agg_impacts = {}
    total_modeled_drop = 0
    
    # PASS 1: Calculate all models and store results
    for city in dropping_cities[:4]:  # Top 4 dropping cities
        cd_f2 = sku_df[sku_df["City"] == city].copy()
        city_prefix = city.split('-')[0][:6].lower()
        cd_f1 = df1_sku[df1_sku[city_col_f1].str.lower().str.contains(city_prefix, na=False)].copy() if city_col_f1 else pd.DataFrame()
        cd_comp = comp[comp["City"].str.lower().str.contains(city_prefix, na=False)].copy()
        
        if cd_f2.empty: continue
        
        # Build Daily TS for City
        ts = cd_f2.groupby("Date").agg(
            Revenue=("Offtake MRP","sum"),
            Wt_OSA=("Wt. OSA %","mean"),
            Discount=("Wt. Discount %","mean"),
            Ad_SOV=("Ad SOV","mean"),
            Organic_SOV=("Organic SOV","mean"),
        ).reset_index().sort_values("Date").reset_index(drop=True)
        
        # Competitor signals
        if not cd_comp.empty:
            comp_ts = cd_comp.groupby("Date").agg(
                Comp_Discount=("Wt. Discount %","mean"),
                Comp_Ad_SOV=("Ad SOV","mean"),
                Comp_OSA=("Wt. OSA %","mean"),
                Comp_Organic_SOV=("Organic SOV","mean"),
            ).reset_index()
            ts = ts.merge(comp_ts, on="Date", how="left")
            ts["Comp_Disc_Adv"] = ts["Comp_Discount"] - ts["Discount"]
        
        # Store signals
        if not cd_f1.empty:
            st = cd_f1.groupby("Date").agg(
                Stores=("Store ID","nunique"),
                Store_OSA=("Avg. OSA %","mean"),
                Stock=("SKU Stock Levels","mean"),
                Listing=("Listing %","mean"),
            ).reset_index()
            ts = ts.merge(st, on="Date", how="left")
            
            # HV split
            if "Locality Sales Contribution" in cd_f1.columns:
                thr = cd_f1["Locality Sales Contribution"].quantile(0.75)
                cd_f1["IsHV"] = (cd_f1["Locality Sales Contribution"] >= thr).astype(int)
                for gv, gn in [(1,"HV"),(0,"LV")]:
                    g = cd_f1[cd_f1["IsHV"]==gv]
                    if not g.empty:
                        gts = g.groupby("Date").agg(OSA=("Avg. OSA %","mean"), Stock=("SKU Stock Levels","mean")).reset_index()
                        gts.columns = ["Date", f"{gn}_OSA", f"{gn}_Stock"]
                        ts = ts.merge(gts, on="Date", how="left")
        
        # Derived signals
        if "Stock" in ts.columns:
            ts["Stock_roll3"] = ts["Stock"].rolling(3, min_periods=2).mean()
        if "Comp_Disc_Adv" in ts.columns:
            ts["Comp_Squeeze"] = ts["Comp_Disc_Adv"] * (100 - ts["Wt_OSA"]) / 100
        ts["DOW"] = ts["Date"].dt.dayofweek
        ts["Is_Monday"] = (ts["DOW"] == 0).astype(int)
        
        s1r = ts[ts["Date"] == d1]
        s2r = ts[ts["Date"] == d2]
        if s1r.empty or s2r.empty: continue
        s1, s2 = s1r.iloc[0], s2r.iloc[0]
        city_drop = s2["Revenue"] - s1["Revenue"]
        total_modeled_drop += city_drop
        
        sens, selected = _fit_sku_city_regression_dynamic(ts)
        r2_model = sens[selected[0]["feat"]]["r2"] if selected and selected[0]["feat"] in sens else 0.0
        
        # Categorize
        cat_map = {
            "Discount": "Own Levers", "Ad_SOV": "Own Levers", "Wt_OSA": "Own Levers", "Organic_SOV": "Own Levers",
            "Comp_Disc_Adv": "Competitor Pressure", "Comp_Discount": "Competitor Pressure", 
            "Comp_Squeeze": "Competitor Pressure", "Comp_OSA": "Competitor Pressure", "Comp_Ad_SOV": "Competitor Pressure",
            "Stores": "Store Quality", "Store_OSA": "Store Quality", "Stock": "Store Quality", "Listing": "Store Quality",
            "HV_OSA": "Store Quality", "HV_Stock": "Store Quality", "LV_OSA": "Store Quality", "LV_Stock": "Store Quality",
            "Stock_roll3": "Store Quality", "Is_Monday": "Structural"
        }
        
        total_explained = 0
        city_rows = []
        for s in selected:
            feat = s["feat"]
            if feat not in sens: continue
            coef = sens[feat]["coef"]
            lag = s["lag"]
            
            dt1 = d1 - pd.Timedelta(days=lag)
            dt2 = d2 - pd.Timedelta(days=lag)
            
            val1_row = ts[ts["Date"] == dt1]
            val2_row = ts[ts["Date"] == dt2]
            
            v1 = val1_row.iloc[0][feat] if not val1_row.empty and not pd.isna(val1_row.iloc[0][feat]) else 0.0
            v2 = val2_row.iloc[0][feat] if not val2_row.empty and not pd.isna(val2_row.iloc[0][feat]) else 0.0
            
            delta = v2 - v1
            impact = delta * coef
            if abs(impact) < 50: continue  # Skip negligible impacts in display
            
            total_explained += impact
            
            category_str = cat_map.get(feat, "Other")
            feat_display = f"{feat} (Lag {lag}d)" if lag > 0 else feat
            
            # Aggregate for overall table
            agg_key = feat_display
            if agg_key not in agg_impacts:
                agg_impacts[agg_key] = {"impact": 0, "feat": feat}
            agg_impacts[agg_key]["impact"] += impact
            
            if "SOV" in feat or "Discount" in feat or "OSA" in feat or "Listing" in feat:
                delta_str = f"{delta:+.1f}%"
                v1_str, v2_str = f"{v1:.1f}%", f"{v2:.1f}%"
            else:
                delta_str = f"{delta:+.2f}"
                v1_str, v2_str = f"{v1:.2f}", f"{v2:.2f}"
                
            insight = f"Regression coef: Rs.{coef:,.0f} per unit."
            if feat == "Is_Monday": insight = "Structural day-of-week demand drop."
            if feat == "Comp_Disc_Adv": insight = "Relative pricing swing vs competitors."
            if feat == "Comp_Squeeze": insight = "Dual pressure of low OSA and competitor discount."
            
            impact_str = f"**Rs.{impact:+,.0f}**" if impact < 0 else f"*Rs.{impact:+,.0f}*"
            row_str = f"| **{category_str}** | {feat_display} | {v1_str} | {v2_str} | **{delta_str}** | {impact_str} | {insight} |"
            city_rows.append(row_str)
            
        unexplained = city_drop - total_explained
        if "Market Fluctuation (Unexplained)" not in agg_impacts:
            agg_impacts["Market Fluctuation (Unexplained)"] = {"impact": 0, "feat": "Market Fluctuation"}
        agg_impacts["Market Fluctuation (Unexplained)"]["impact"] += unexplained
        
        unexplained_str = f"**Rs.{unexplained:+,.0f}**" if unexplained < 0 else f"*Rs.{unexplained:+,.0f}*"
        row_str = f"| **Unexplained** | Market Fluctuation | - | - | - | {unexplained_str} | Natural platform variance / noise. |"
        city_rows.append(row_str)
        
        city_results.append({
            "city": city,
            "drop": city_drop,
            "r2": r2_model,
            "signals": len(ts.columns)-3,
            "rows": city_rows
        })
        
    # PASS 2: Build the Markdown Output
    markdown_lines = []
    markdown_lines.append(f"#### 📦 {sku_name}")
    markdown_lines.append(f"**Total Revenue:** Rs.{total_rev_d1:,.0f} -> Rs.{total_rev_d2:,.0f} (Drop: **Rs.{total_drop:+,.0f}**)")
    markdown_lines.append("")
    
    # Build City Drop Summary Table
    markdown_lines.append("**1. City Drop Summary**")
    markdown_lines.append("")
    markdown_lines.append("| City | Revenue Drop (Rs.) |")
    markdown_lines.append("| :--- | :--- |")
    for cr in city_results:
        drop_str = f"**Rs.{cr['drop']:,.0f}**" if cr['drop'] < 0 else f"*Rs.{cr['drop']:,.0f}*"
        markdown_lines.append(f"| **{cr['city']}** | {drop_str} |")
        
    unmodeled = total_drop - total_modeled_drop
    unmodeled_str = f"*Rs.{unmodeled:,.0f}*"
    markdown_lines.append(f"| *Unmodeled Smaller Cities* | {unmodeled_str} |")
    markdown_lines.append(f"| **Total SKU Drop** | **Rs.{total_drop:,.0f}** |")
    markdown_lines.append("")
    
    # Build Aggregated Driver Impact Table
    markdown_lines.append("**2. Aggregated Driver Impact (Across Modeled Cities)**")
    markdown_lines.append("")
    markdown_lines.append("| Metric / Driver | Total Rs. Impact | Insight |")
    markdown_lines.append("| :--- | :--- | :--- |")
    
    sorted_agg = sorted(agg_impacts.items(), key=lambda x: x[1]["impact"])
    for name, data in sorted_agg:
        imp = data["impact"]
        if abs(imp) < 100: continue
        feat = data["feat"]
        
        insight = ""
        if "Market Fluctuation" in name: insight = "Natural platform variance / noise."
        elif "Is_Monday" in name: insight = "Natural day-of-week drop."
        elif "Ad_SOV" in name: insight = "Ad spend visibility impact."
        elif "Organic_SOV" in name: insight = "Organic search ranking impact."
        elif "Discount" in name and "Comp" not in name: insight = "Direct impact of own discount."
        elif "Comp_Discount" in name: insight = "Competitor pricing changes."
        elif "Comp_Disc_Adv" in name: insight = "Relative pricing gap vs competitors."
        elif "OSA" in name and "Comp" not in name: insight = "Own stock availability issues."
        elif "Comp_OSA" in name: insight = "Competitor availability changes."
        elif "Stock" in name: insight = "Impact of total SKU inventory in darkstores."
        elif "Listing" in name: insight = "Store assortment/listing gap."
        elif "Squeeze" in name: insight = "Dual pressure from low OSA and competitor discount."
        
        imp_str = f"**Rs.{imp:+,.0f}**" if imp < 0 else f"*Rs.{imp:+,.0f}*"
        markdown_lines.append(f"| **{name}** | {imp_str} | {insight} |")
        
    markdown_lines.append("")
    markdown_lines.append("---")
    markdown_lines.append("")
    
    # Append individual city tables
    for cr in city_results:
        markdown_lines.append(f"#### 🏙️ CITY: {cr['city']} | Drop: Rs.{cr['drop']:+,.0f}")
        markdown_lines.append(f"**City Model Accuracy (R²):** {cr['r2']:.2f} | **Signals Tested:** {cr['signals']}")
        markdown_lines.append("")
        markdown_lines.append("| Driver Category | Metric | Apr 19 | Apr 20 | Delta | Impact | Insight |")
        markdown_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        markdown_lines.extend(cr['rows'])
        markdown_lines.append("")
        
    return {"sku_name": sku_name, "rev_d1": total_rev_d1, "rev_d2": total_rev_d2, 
            "rev_delta": total_drop, "worst_city": c_deltas.index[0] if len(c_deltas)>0 else "Unknown",
            "worst_city_drop": c_deltas.iloc[0] if len(c_deltas)>0 else 0,
            "markdown_table": "\\n".join(markdown_lines)}


# ── DATE COMPARISON ENGINE (Apr19 vs Apr20) ───────────────────────────────────
def compare_two_days(df24, df_all, df1, ts, d1_str, d2_str, lag):
    print(f"[COMPARE] {d1_str} vs {d2_str} deep-dive ...")
    d1 = pd.Timestamp(d1_str)
    d2 = pd.Timestamp(d2_str)

    def day_agg(df, d):
        s = df[df["Date"]==d]
        if s.empty: return {}
        return {"Revenue":float(s["Offtake MRP"].sum()),
                "Units":float(s["Units"].sum()) if "Units" in s else None,
                "OSA":float(s["Wt. OSA %"].mean()),
                "Discount":float(s["Wt. Discount %"].mean()),
                "Ad_SOV":float(s["Ad SOV"].mean()),
                "Organic_SOV":float(s["Organic SOV"].mean()),
                "Overall_SOV":float(s["Overall SOV"].mean()),
                "Cat_Share":float(s["Category Share"].mean()),
                "Darkstores":float(s["Darkstore_Count"].sum()),
                "Network_Strength":float(s["Network_Strength"].sum()),
                "SKUs_active":int(s["Product ID"].nunique()),
                "Cities_active":int(s["City"].nunique())}

    s1 = day_agg(df24, d1)
    s2 = day_agg(df24, d2)
    if not s1 or not s2:
        return {"error": f"No data for {d1_str} or {d2_str}"}

    # Delta for every metric
    deltas = {}
    for k in s1:
        if s1[k] is not None and s2[k] is not None:
            delta = s2[k]-s1[k]
            pct   = delta/s1[k]*100 if s1[k]!=0 else 0
            deltas[k] = {"d1":round(s1[k],2),"d2":round(s2[k],2),
                         "delta":round(delta,2),"pct":round(pct,1)}

    # City-level
    city1 = df24[df24["Date"]==d1].groupby("City").agg(
                Rev=("Offtake MRP","sum"),OSA=("Wt. OSA %","mean"),
                Disc=("Wt. Discount %","mean"),SOV=("Ad SOV","mean"),
                Dark=("Network_Strength","sum")).reset_index()
    city2 = df24[df24["Date"]==d2].groupby("City").agg(
                Rev=("Offtake MRP","sum"),OSA=("Wt. OSA %","mean"),
                Disc=("Wt. Discount %","mean"),SOV=("Ad SOV","mean"),
                Dark=("Network_Strength","sum")).reset_index()
    city_merged = city1.merge(city2,on="City",suffixes=("_d1","_d2"),how="outer").fillna(0)
    city_merged["Rev_Delta"]  = city_merged["Rev_d2"]-city_merged["Rev_d1"]
    city_merged["OSA_Delta"]  = city_merged["OSA_d2"]-city_merged["OSA_d1"]
    city_merged["Dark_Delta"] = city_merged["Dark_d2"]-city_merged["Dark_d1"]
    city_merged["Rev_Pct"]    = city_merged.apply(
        lambda r: r["Rev_Delta"]/r["Rev_d1"]*100 if r["Rev_d1"]!=0 else 0, axis=1)

    city_detail = city_merged.sort_values("Rev_Delta")[
        ["City","Rev_d1","Rev_d2","Rev_Delta","Rev_Pct","OSA_d2","OSA_Delta","Dark_d2","Dark_Delta"]
    ].round(1).to_dict(orient="records")

    # SKU-level top losers
    sku1 = df24[df24["Date"]==d1].groupby(["Product ID","Product Name","Grammage"])["Offtake MRP"].sum().reset_index(name="Rev_d1")
    sku2 = df24[df24["Date"]==d2].groupby(["Product ID","Product Name","Grammage"])["Offtake MRP"].sum().reset_index(name="Rev_d2")
    sku  = sku1.merge(sku2,on=["Product ID","Product Name","Grammage"],how="outer").fillna(0)
    sku["Delta"] = sku["Rev_d2"]-sku["Rev_d1"]
    sku["Pct"]   = sku.apply(lambda r: r["Delta"]/r["Rev_d1"]*100 if r["Rev_d1"]!=0 else 0,axis=1)
    top_sku_losers  = sku.nsmallest(10,"Delta")[["Product ID","Product Name","Grammage","Rev_d1","Rev_d2","Delta","Pct"]].round(1).to_dict(orient="records")
    top_sku_gainers = sku.nlargest(10,"Delta")[["Product ID","Product Name","Grammage","Rev_d1","Rev_d2","Delta","Pct"]].round(1).to_dict(orient="records")

    # Category competitive — same correct logic as L6
    # Group by (City, Category) to avoid double-counting multiple SKUs in same category pool
    def day_cat_rev(df24_day):
        valid = df24_day[
            df24_day["Offtake MRP"].notna() &
            df24_day["Category Share"].notna() &
            (df24_day["Category Share"] > 0)
        ].copy()
        # Category Share is already % (e.g. 6.46 = 6.46%)
        valid["Implied_Cat_Rev"] = valid["Offtake MRP"] / (valid["Category Share"] / 100)
        grp = valid.groupby(["City", "Category"]).agg(
            brand_rev=("Offtake MRP", "sum"),
            cat_rev=("Implied_Cat_Rev", "mean"),  # mean per group avoids double-count
        ).reset_index()
        total_brand = grp["brand_rev"].sum()
        total_cat   = grp["cat_rev"].sum()
        share_pct   = total_brand / total_cat * 100 if total_cat > 0 else 0
        return float(total_cat), float(share_pct)

    cat1_rev, share1_pct = day_cat_rev(df24[df24["Date"] == d1])
    cat2_rev, share2_pct = day_cat_rev(df24[df24["Date"] == d2])
    brand_share1 = share1_pct
    brand_share2 = share2_pct
    cat_change   = (cat2_rev - cat1_rev) / cat1_rev * 100 if cat1_rev > 0 else 0
    brand_change = deltas.get("Revenue", {}).get("pct", 0)
    share_chg    = round(brand_share2 - brand_share1, 2)


    # Attribution: which drivers caused how much of the revenue delta
    rev_delta = s2["Revenue"]-s1["Revenue"]
    drivers_list = ["OSA","Discount","Ad_SOV","Network_Strength","Organic_SOV"]
    
    # Day-of-Week structural effect (e.g. Sunday -> Monday naturally drops)
    dow1 = d1.dayofweek
    dow2 = d2.dayofweek
    dow_avg = ts.groupby(ts["Date"].dt.dayofweek)["Revenue"].mean()
    dow_expected_d1 = dow_avg.get(dow1, float("nan"))
    dow_expected_d2 = dow_avg.get(dow2, float("nan"))
    dow_structural_effect = (dow_expected_d2 - dow_expected_d1) if not (pd.isna(dow_expected_d1) or pd.isna(dow_expected_d2)) else 0.0
    
    sens = _fit_sku_city_regression(ts, drivers_list)
    
    drv_attribution = []
    for d in drivers_list:
        dcol = d
        dd  = (s2.get(dcol,0) or 0)-(s1.get(dcol,0) or 0)
        coef = sens[d]["coef"]
        imp = dd * coef
        lag_d = lag.get(d,{}).get("best_lag_days",0)
        note  = f"lag={lag_d}d -- impact felt {'immediately' if lag_d==0 else f'{lag_d} day(s) after change'}"
        
        if imp != 0:
            hist_ev = (f"Multivariate OLS over 20 days controls for confounders. "
                       f"Every 1 unit change in {d} explains Rs. {coef:,.0f} impact independently. "
                       f"Today's {dd:.2f} delta * {coef:,.0f} = Rs. {imp:,.0f}")
        else:
            hist_ev = "N/A - Impact nullified."
        
        driver_name = d
        custom_narrative = None
        if d == "Network_Strength":
            dark_dd = (s2.get("Darkstores",0) or 0)-(s1.get("Darkstores",0) or 0)
            net_delta = dd
            base_val = s1.get(dcol,0) or 0
            curr_val = s2.get(dcol,0) or 0
            store_action = f"lost {abs(dark_dd):.0f}" if dark_dd < 0 else (f"gained {dark_dd:.0f}" if dark_dd > 0 else "saw no change in")
            cap_action = f"wiped out {abs(net_delta):.1f}%" if net_delta < 0 else (f"added {net_delta:.1f}%" if net_delta > 0 else "didn't change")
            imp_action = f"causing a drop of Rs. {abs(imp):,.0f}" if imp < 0 else f"driving a gain of Rs. {abs(imp):,.0f}"
            custom_narrative = f"You {store_action} physical stores, which {cap_action} of your sales capacity ({base_val:.1f}% -> {curr_val:.1f}%), {imp_action}."
            driver_name = "Network_Capacity"

        drv_attribution.append({"driver":driver_name,"d1_value":round(s1.get(dcol,0),2),
            "d2_value":round(s2.get(dcol,0),2),"delta":round(dd,2),
            "revenue_impact_rs":round(imp,0),"share_pct":round(imp/rev_delta*100,1) if rev_delta!=0 else 0,
            "lag_note":note, "historical_evidence": hist_ev, "custom_narrative": custom_narrative})
    drv_attribution.sort(key=lambda x:abs(x["revenue_impact_rs"]),reverse=True)

    dir_word = "DECLINE" if rev_delta < 0 else "GROWTH"
    pct_word  = abs(round(brand_change, 1))
    cat_word  = "fell" if cat_change < 0 else "rose"
    if brand_change < 0 and cat_change < 0 and abs(cat_change) >= abs(brand_change) * 0.7:
        verdict = "MARKET_ISSUE"
    elif brand_change < 0 and abs(cat_change) < 2:
        verdict = "OWN_ISSUE"   # brand fell, category flat
    elif brand_change < cat_change:
        verdict = "OWN_ISSUE"
    else:
        verdict = "OUTPERFORMING"
    top_drv = drv_attribution[0] if drv_attribution else {}

    narrative_lines = [
        f"### === {BRAND_DISPLAY} | {d1_str} vs {d2_str} ===",
        "",
        f"**HEADLINE**: Revenue {dir_word} of {pct_word}% (Rs.{rev_delta:+,.0f})",
        f"- {d1_str}: Rs.{s1['Revenue']:,.0f}  |  {d2_str}: Rs.{s2['Revenue']:,.0f}",
        "",
        f"**MARKET CONTEXT**:",
        f"- Blinkit organic category revenue {cat_word} {abs(cat_change):.1f}% on same day.",
        f"- {BRAND_DISPLAY} share: {brand_share1:.1f}% -> {brand_share2:.1f}% ({share_chg:+.1f} pts)",
        f"- Verdict: **{verdict}**",
        ("  => The overall category also declined. Market-level headwind."
         if verdict == "MARKET_ISSUE"
         else "  => Category was flat but brand fell -- OWN levers are the root cause."
         if verdict == "OWN_ISSUE" and abs(cat_change) < 2
         else "  => Brand underperformed the category -- own operational/marketing levers are the root cause."
         if verdict == "OWN_ISSUE"
         else "  => Brand grew faster than category -- maintain what is working."),
        "",
        "**DRIVER SNAPSHOT (d1 -> d2)**:",
    ]
    for k in ["OSA","Discount","Ad_SOV","Darkstores","SKUs_active","Cities_active"]:
        if k in deltas:
            d=deltas[k]
            arrow = "v" if d["delta"]<0 else ("^" if d["delta"]>0 else "=")
            narrative_lines.append(f"- **{k}**: {d['d1']} -> {d['d2']}  ({arrow} {d['pct']:+.1f}%)")
    narrative_lines += [
        "",
        "**TOP FINANCIAL ATTRIBUTION (what caused the revenue move)**:",
    ]
    for a in drv_attribution:
        sign = "+" if a["revenue_impact_rs"]>=0 else ""
        if a.get("custom_narrative"):
            narrative_lines.append(f"- **{a['driver']}**: {a['custom_narrative']}")
        else:
            narrative_lines.append(
                f"- **{a['driver']}**: {sign}Rs.{a['revenue_impact_rs']:,.0f} ({a['share_pct']}% of delta)  {a['lag_note']}")
        if a.get("historical_evidence"):
            narrative_lines.append(f"  > *{a['historical_evidence']}*")
    
    # Day-of-Week structural baseline effect
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    if abs(dow_structural_effect) > 100:
        dow_sign = "+" if dow_structural_effect >= 0 else ""
        narrative_lines.append(f"- **Day-of-Week Baseline Effect** ({day_names[dow1]} -> {day_names[dow2]}): {dow_sign}Rs.{dow_structural_effect:,.0f}")
        narrative_lines.append(f"  > *Structural: historically {day_names[dow2]} averages Rs.{dow_structural_effect:,.0f} {'more' if dow_structural_effect > 0 else 'less'} than {day_names[dow1]}. Not an operational issue.*")
    else:
        dow_structural_effect = 0.0
            
    total_explained = sum(a['revenue_impact_rs'] for a in drv_attribution) + dow_structural_effect
    unexplained_top = rev_delta - total_explained
    if abs(unexplained_top) > 100:
        narrative_lines.append(f"- **Unexplained / Organic**: Rs.{unexplained_top:,.0f} ({(unexplained_top/rev_delta)*100:.1f}% of delta)")
        narrative_lines.append(f"  > *This portion cannot be statistically linked to any measured operational driver or day-of-week pattern.*")
        
    narrative_lines += [
        "",
        "**GEOGRAPHIC STORY (biggest city movers)**:",
    ]
    for row in sorted(city_detail, key=lambda x:x["Rev_Delta"])[:5]:
        narrative_lines.append(
            f"- **{row['City']}**: Rs.{row['Rev_Delta']:+.0f} | OSA={row['OSA_d2']:.1f}%"
            f" (OSA chg {row['OSA_Delta']:+.1f}%) | Net_Strength={row['Dark_d2']:.0f} (chg {row['Dark_Delta']:+.0f})")
    narrative_lines += ["", "**TOP LOSING SKUs**:", ""]
    for s in top_sku_losers[:5]:
        narrative_lines.append(
            f"- {s['Product Name']} {s['Grammage']}: Rs.{s['Delta']:+.0f} ({s['Pct']:+.1f}%)")

    sku_deep_dives = []
    narrative_lines += ["", "### SKU DEEP DIVE (Top Losers with Historical Validation)"]
    for s in top_sku_losers[:5]:
        sd = sku_deep_dive(df24, df_all, df1, s["Product ID"], f"{s['Product Name']} {s['Grammage']}", d1_str, d2_str)
        if not sd: continue
        sku_deep_dives.append(sd)
        if "markdown_table" in sd:
            narrative_lines.extend(sd["markdown_table"].split('\\n'))

    # ── ATTRIBUTION SUMMARY TABLE ────────────────────────────────────────────────
    return {
        "comparison_dates": {"d1":d1_str,"d2":d2_str},
        "headline_delta_rs": round(rev_delta,0),
        "headline_pct":      round(brand_change,1),
        "market_verdict":    verdict,
        "day_summaries":     {"d1":s1,"d2":s2},
        "metric_deltas":     deltas,
        "driver_attribution":drv_attribution,
        "city_detail":       city_detail,
        "top_sku_losers":    top_sku_losers,
        "top_sku_gainers":   top_sku_gainers,
        "sku_deep_dives":    sku_deep_dives,
        "narrative":         "\n".join(narrative_lines),
        "category_context":  {"cat_revenue_d1":round(float(cat1_rev),0),"cat_revenue_d2":round(float(cat2_rev),0),
                               "cat_change_pct":round(cat_change,2),
                               "brand_share_d1":round(brand_share1,2),"brand_share_d2":round(brand_share2,2),
                               "share_point_change":round(share_chg,2)},
    }


# ── LLM BRIEF GENERATOR ───────────────────────────────────────────────────────
def generate_brief(report, outdir):
    sep  = "=" * 70
    L    = []
    comp = report.get("COMPARISON", {})
    l0   = report.get("L0_raw_snapshot", {})
    l1   = report.get("L1_baseline", {})
    l4   = report.get("L4_lag", {})
    l3   = report.get("L3_thresholds", {})
    l2   = report.get("L2_interactions", {})
    l5   = report.get("L5_pullforward", {})
    l6   = report.get("L6_competitive", {})
    l7   = report.get("L7_geographic", {})
    l8   = report.get("L8_leading", {})
    l9   = report.get("L9_attribution", {})
    l10  = report.get("L10_feedback", {})

    L += [f"# ADVANCED SALES RCA -- {BRAND_DISPLAY} (Blinkit) - FINAL EXECUTIVE SUMMARY",
          f"**Generated** : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
          f"**Period**    : {report['meta']['date_range']}  ({report['meta']['total_days']} days)",
          "---", ""]

    # L0 Raw Snapshot (Evidence First)
    if l0:
        L += ["## L0 -- RAW DAILY SNAPSHOT (Evidence Base)",
              "| Date | DOW | Revenue | OSA | Network Strength | Ad SOV | Discount |",
              "|---|---|---|---|---|---|---|"]
        for r in l0.get("last_n_days", []):
            rev_str = f"Rs.{r['revenue_rs']:,.0f} ({r['revenue_L']})"
            L.append(f"| {r['date']} | {r['dow']} | {rev_str} | {str(r['OSA'])} | {str(r.get('Network_Strength', r.get('Darkstores')))} | {str(r['Ad_SOV'])} | {str(r['Discount'])} |")
        L.append("")

    # Comparison
    if comp and "narrative" in comp:
        L += [f"## PRIMARY COMPARISON: {comp['comparison_dates']['d1']} vs {comp['comparison_dates']['d2']}", ""]
        L.append(comp["narrative"])
        L.append("")

    # L4
    L.append("## L4 -- LAG ANALYSIS (Cause & Effect Timing)")
    for d, info in l4.items():
        sig = "significant" if info.get("significant") else "NOT significant"
        L.append(f"- **{d}**: lag={info.get('best_lag_days')}d | r={info.get('best_r')} [{sig}] {info.get('direction')}")
        if info.get("lag_validation"):
            for ex in info["lag_validation"]:
                L.append(f"  - *{ex}*")
    L.append("")

    # L3
    L.append("## L3 -- THRESHOLD ZONES")
    for d, info in l3.items():
        zone = f"[{info.get('current_zone','')}]"
        L.append(f"- **{d}**: {info.get('current')} {zone} (STATUS=**{info.get('status')}**)")
        if info.get("status") in ("CRITICAL", "WARNING"):
            L.append(f"  - ⚠️ Sales penalty at breach: {info.get('sales_penalty_pct')}%")
        if info.get("zone_bands"):
            for lo, hi, zname, zdesc in info["zone_bands"]:
                L.append(f"  - {zdesc}")
        if info.get("last_3_days"):
            trend_str = " | ".join([f"{t['date']}={t['value']}" for t in info["last_3_days"]])
            L.append(f"  - Trend: {trend_str}")
    L.append("")

    # L2
    L.append("## L2 -- DRIVER INTERACTIONS")
    for w in l2.get("warnings",[]): L.append(f"- 🔴 {w}")
    for p in l2.get("positives",[]): L.append(f"- 🟢 {p}")
    if not l2.get("warnings") and not l2.get("positives"): L.append("- No critical interactions.")
    L.append("")

    # L1
    L += ["## L1 -- SMART BASELINE",
          f"- **Trend slope**: Rs.{l1.get('trend_slope_day','N/A')}/day",
          f"- **Trend R2**: {l1.get('trend_r2','N/A')}",
          f"- **True anomaly dates**: {l1.get('true_anomaly_dates',[])}", ""]

    # L6
    L += ["## L6 -- MARKET vs OWN FACTOR",
          f"- **Verdict**: **{l6.get('latest_verdict','N/A')}**",
          f"- {l6.get('latest_explanation','')}",
          f"- Brand DoD: {l6.get('brand_dod')}% | Category DoD: {l6.get('cat_dod')}%", ""]

    # L7
    L += ["## L7 -- GEOGRAPHIC CASCADE",
          f"- **National change**: Rs.{l7.get('total_national_change',0):,.0f}",
          f"- **Pattern**: {l7.get('drop_pattern')} -- {l7.get('drop_pattern_note','')}",
          "", "### Top losing cities:"]
    for c in l7.get("top_losing_cities",[]): L.append(f"- **{c['City']}** ({c['Tier']}): Rs.{c['Change']:+.0f} | OSA={c['OSA']:.1f}% | Net_Strength={c['Network_Strength']:.1f}")
    L.append("")

    # L8
    L.append("## L8 -- LEADING INDICATOR ALERTS")
    if not l8.get("alerts"): L.append("- ✅ No deteriorating trends.")
    for a in l8.get("alerts",[]):
        L += [f"- **[{a['severity']}]** {a['alert']}",
              f"  - Trend: {a.get('daily_trend')} (net {a.get('net_change'):+.2f})",
              f"  - => *{a['prediction']}*"]
    L.append("")

    # L9
    L += ["## L9 -- FINANCIAL ATTRIBUTION",
          f"- {l9.get('baseline_date')} -> {l9.get('analysis_date')}",
          f"- **Total Rev Delta**: Rs.{l9.get('total_delta_rs',0):,.0f}"]
    for d in l9.get("driver_attributions",[]):
        s = "+" if d["revenue_impact_rs"]>=0 else ""
        if d.get("custom_narrative"):
             L.append(f"- **{d['driver']}**: {d['custom_narrative']} | lag={d['lag_days']}d")
        else:
             L.append(f"- **{d['driver']}**: {s}Rs.{d['revenue_impact_rs']:,.0f} ({d['share_pct']}%) | lag={d['lag_days']}d")
        if d.get("historical_evidence"):
            L.append(f"  > *{d['historical_evidence']}*")
    L.append("")

    # L10
    L += ["## L10 -- CONFIDENCE & FEEDBACK",
          f"- **Score**  : {l10.get('confidence_score')}%",
          f"- **Level**  : {l10.get('interpretation')}",
          f"- **Guardrail**: *{l10.get('guardrail_message')}*"]
    if l10.get("reasons_for_score"):
        L.append("- **Reasons**:")
        for r in l10["reasons_for_score"]: L.append(f"  - {r}")
    if l10.get("recommended_actions"):
        L.append("- **Actions**:")
        for a in l10["recommended_actions"]: L.append(f"  - {a}")
    L += [f"- **Override**: {l10.get('override_instructions','')}", "",
          "---", "**END OF FINAL EXECUTIVE SUMMARY**", "---"]

    text = "\n".join(L)
    path = outdir / "advanced_rca_executive_summary.md"
    path.write_text(text, encoding="utf-8")
    return text, path


# ── MAIN ORCHESTRATOR ─────────────────────────────────────────────────────────
def main():
    out = make_output_dir()
    print(f"\n[OUTPUT] Folder: {out}\n")

    df1, df24, df_all, active_stores = load_data()
    ts = build_ts(df24)
    print(f"\nDate range: {ts['Date'].min().date()} to {ts['Date'].max().date()}  | {len(ts)} days\n")
    
    # --- LLM Evidence Pack Generation ---
    loc_stats = build_locality_intelligence(active_stores)
    sku_city_ts = build_sku_city_ts(df24, loc_stats)
    sku_city_ts = enrich_sku_city_ts(sku_city_ts)

    l0       = L0_raw_snapshot(ts)
    l1, ts   = L1_baseline(ts)
    l4       = L4_lag(ts)
    l3       = L3_thresholds(ts, l4)
    l2       = L2_interactions(ts)
    l5       = L5_pullforward(ts)
    l6       = L6_competitive(df24, df_all)
    l7       = L7_geographic(df24, l6.get("latest_verdict", ""))
    l8       = L8_leading(ts, df24)
    l9       = L9_attribution(ts, l4)
    l10      = L10_feedback(l1, l9, l4, len(ts))
    comp     = compare_two_days(df24, df_all, df1, ts, COMPARE_DATE1, COMPARE_DATE2, l4)
    attr_day = build_sku_city_attribution(sku_city_ts, df24, df_all, COMPARE_DATE1, COMPARE_DATE2, mode="day", nat_lag=l4)
    attr_period = build_sku_city_attribution(sku_city_ts, df24, df_all, COMPARE_PERIOD1_START, COMPARE_PERIOD1_END, mode="period", nat_lag=l4)
    
    export_llm_evidence_pack(df24, df1, ts, loc_stats, sku_city_ts, l6, l9, comp, attr_day, attr_period, out)

    report = {
        "meta": {"brand":BRAND_DISPLAY,"generated_at":datetime.now().isoformat(),
                 "date_range":f"{ts['Date'].min().date()} to {ts['Date'].max().date()}",
                 "total_days":len(ts)},
        "L0_raw_snapshot":l0,
        "L1_baseline":l1,"L2_interactions":l2,"L3_thresholds":l3,"L4_lag":l4,
        "L5_pullforward":l5,"L6_competitive":l6,"L7_geographic":l7,
        "L8_leading":l8,"L9_attribution":l9,"L10_feedback":l10,
        "COMPARISON":comp,
    }

    json_path = out / "advanced_rca_report.json"
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OK] JSON report  -> {json_path}")

    _, brief_path = generate_brief(report, out)
    print(f"[OK] Executive Summary -> {brief_path}")

    # Print the comparison narrative directly to console
    print("\n" + "="*70)
    print(comp.get("narrative","No comparison generated."))
    print("="*70)
    print(f"\nConfidence: {l10['confidence_score']}%  |  Verdict: {l6['latest_verdict']}")
    print(f"Drop pattern: {l7['drop_pattern']}  |  Top driver: {l9.get('top_driver')}")
    print(f"Leading alerts: {l8['total_alerts']}")
    print(f"\n[OUTPUT FOLDER] {out}")



def build_locality_intelligence(active_stores):
    print("[LLM Evidence] Building locality intelligence from raw stores ...")
    # Group by Date, City, Product ID to get locality statistics
    loc_stats = active_stores.groupby(["Date", "city_key", "Product ID"]).agg(
        Darkstore_Count=("Store ID", "size"),
        Avg_Locality_Score=("Locality Sales Contribution", "mean"),
        Max_Locality_Score=("Locality Sales Contribution", "max"),
        Min_Locality_Score=("Locality Sales Contribution", "min"),
        High_Value_Store_Count=("Is_High_Value", "sum"),
        Network_Strength=("Locality Sales Contribution", "sum")
    ).reset_index()
    
    # Calculate P75 and HHI separately as they require custom functions
    def p75(x):
        return x.quantile(0.75) if len(x) > 0 else 0
        
    def hhi(x):
        total = x.sum()
        if total == 0: return 0
        shares = (x / total) * 100
        return (shares ** 2).sum()

    custom_stats = active_stores.groupby(["Date", "city_key", "Product ID"])["Locality Sales Contribution"].agg([
        ("P75_Locality_Score", p75),
        ("HHI", hhi)
    ]).reset_index()
    
    loc_stats = loc_stats.merge(custom_stats, on=["Date", "city_key", "Product ID"], how="left")
    loc_stats["Low_Value_Store_Count"] = loc_stats["Darkstore_Count"] - loc_stats["High_Value_Store_Count"]
    loc_stats["High_Value_Store_Pct"] = (loc_stats["High_Value_Store_Count"] / loc_stats["Darkstore_Count"] * 100).fillna(0)
    
    return loc_stats

def build_sku_city_ts(df24, loc_stats):
    print("[LLM Evidence] Building SKU x City time-series ...")
    
    # Base SKU x City stats from df24
    base = df24.groupby(["Date", "City", "city_key", "Product ID", "Product Name", "Grammage"]).agg(
        Revenue=("Offtake MRP", "sum"),
        OSA=("Wt. OSA %", "mean"),
        Discount=("Wt. Discount %", "mean"),
        Ad_SOV=("Ad SOV", "mean"),
        Organic_SOV=("Organic SOV", "mean"),
        Category_Share=("Category Share", "mean")
    ).reset_index()
    
    # Determine tier
    base["Tier"] = base["City"].str.lower().str.strip().apply(lambda c: "Metro" if c in METRO else "Tier2/3")
    
    # Merge with locality stats
    full = base.merge(loc_stats, on=["Date", "city_key", "Product ID"], how="left").fillna({
        "Darkstore_Count": 0, "Avg_Locality_Score": 0, "Max_Locality_Score": 0, 
        "Min_Locality_Score": 0, "High_Value_Store_Count": 0, "Network_Strength": 0,
        "P75_Locality_Score": 0, "HHI": 0, "Low_Value_Store_Count": 0, "High_Value_Store_Pct": 0
    })
    
    # Re-scale Network Strength to match existing logic (* 100)
    full["Network_Strength"] = full["Network_Strength"] * 100
    
    # Calculate DoD changes
    full = full.sort_values(["Product ID", "City", "Date"])
    
    for col in ["Revenue", "Network_Strength", "Darkstore_Count", "Avg_Locality_Score"]:
        full[f"{col}_DoD"] = full.groupby(["Product ID", "City"])[col].diff().fillna(0)
        
    # Calculate National Shares
    national_day_sku = full.groupby(["Date", "Product ID"]).agg(
        Nat_Rev=("Revenue", "sum"),
        Nat_Net=("Network_Strength", "sum")
    ).reset_index()
    
    full = full.merge(national_day_sku, on=["Date", "Product ID"], how="left")
    full["City_Revenue_Share_Pct"] = (full["Revenue"] / full["Nat_Rev"] * 100).fillna(0)
    full["City_Network_Share_Pct"] = (full["Network_Strength"] / full["Nat_Net"] * 100).fillna(0)
    
    full = full.drop(columns=["Nat_Rev", "Nat_Net"])
    
    return full

def export_llm_evidence_pack(df24, df1, ts, loc_stats, sku_city_ts, l6, l9, comp, attr_day, attr_period, outdir):
    print("[LLM Evidence] Exporting evidence pack ...")
    
    import pandas as pd
    import json
    
    excel_path = outdir / "llm_evidence_pack.xlsx"
    json_path = outdir / "llm_evidence_pack.json"
    
    # Sheet 1: Daily National
    daily_nat = ts.copy()
    if "daily" in l6:
        l6_df = pd.DataFrame(l6["daily"])
        if not l6_df.empty and "Date_str" in l6_df.columns:
            l6_df["Date"] = pd.to_datetime(l6_df["Date_str"])
            daily_nat = daily_nat.merge(l6_df[["Date", "Total_Cat_Rev", "Brand_Share_Pct", "Cat_DoD", "Brand_DoD"]], on="Date", how="left")
    
    # Sheet 2: SKU x City x Day is exactly sku_city_ts
    
    # Sheet 3: SKU x Day Summary
    sku_day = sku_city_ts.groupby(["Date", "Product ID", "Product Name", "Grammage"]).agg(
        Revenue=("Revenue", "sum"),
        OSA=("OSA", "mean"),
        Discount=("Discount", "mean"),
        Network_Strength=("Network_Strength", "sum"),
        Darkstores=("Darkstore_Count", "sum"),
        Revenue_DoD=("Revenue_DoD", "sum"),
        Network_DoD=("Network_Strength_DoD", "sum")
    ).reset_index()
    
    # Sheet 4: City x Day Summary
    city_day = sku_city_ts.groupby(["Date", "City"]).agg(
        Revenue=("Revenue", "sum"),
        OSA=("OSA", "mean"),
        Network_Strength=("Network_Strength", "sum"),
        Darkstores=("Darkstore_Count", "sum")
    ).reset_index()
    
    # Sheet 5: Comparison
    d1_str, d2_str = comp.get("comparison_dates", {}).get("d1"), comp.get("comparison_dates", {}).get("d2")
    comp_df = pd.DataFrame()
    if d1_str and d2_str:
        d1 = pd.Timestamp(d1_str)
        d2 = pd.Timestamp(d2_str)
        s1 = sku_city_ts[sku_city_ts["Date"] == d1]
        s2 = sku_city_ts[sku_city_ts["Date"] == d2]
        
        comp_df = s1.merge(s2, on=["Product ID", "Product Name", "Grammage", "City"], suffixes=("_d1", "_d2"), how="outer").fillna(0)
        comp_df["Revenue_Delta"] = comp_df["Revenue_d2"] - comp_df["Revenue_d1"]
        comp_df["Network_Delta"] = comp_df["Network_Strength_d2"] - comp_df["Network_Strength_d1"]
        comp_df["Store_Delta"] = comp_df["Darkstore_Count_d2"] - comp_df["Darkstore_Count_d1"]
        comp_df["Locality_Delta"] = comp_df["Avg_Locality_Score_d2"] - comp_df["Avg_Locality_Score_d1"]
        
    # Sheet 6: Locality Intelligence is sku_city_ts with specific columns
    loc_intel_cols = ["Date", "Product ID", "Product Name", "City", "High_Value_Store_Count", "Low_Value_Store_Count", 
                      "Max_Locality_Score", "Min_Locality_Score", "P75_Locality_Score", "HHI", "High_Value_Store_Pct"]
    loc_intel = sku_city_ts[loc_intel_cols].copy()
    
    # Sheet 7: Attributions
    attr_df = pd.DataFrame(l9.get("driver_attributions", []))
    
    # Write to Excel
    with pd.ExcelWriter(excel_path) as writer:
        daily_nat.to_excel(writer, sheet_name="daily_national", index=False)
        sku_city_ts.to_excel(writer, sheet_name="sku_city_day", index=False)
        sku_day.to_excel(writer, sheet_name="sku_day_summary", index=False)
        city_day.to_excel(writer, sheet_name="city_day_summary", index=False)
        comp_df.to_excel(writer, sheet_name="sku_city_comparison", index=False)
        loc_intel.to_excel(writer, sheet_name="locality_intelligence", index=False)
        if not attr_df.empty:
            attr_df.to_excel(writer, sheet_name="driver_attributions", index=False)
        if not attr_day.empty:
            attr_day.to_excel(writer, sheet_name="sku_city_attr_day", index=False)
        if not attr_period.empty:
            attr_period.to_excel(writer, sheet_name="sku_city_attr_period", index=False)
            
    # Write to JSON
    json_data = {
        "meta": {
            "llm_prompt_hint": "Analyze the sku_city_day and locality_intelligence data to find which SKUs lost high-value darkstores, and which cities are underperforming despite high network capacity."
        },
        "daily_national": daily_nat.assign(Date=daily_nat["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "sku_city_day": sku_city_ts.assign(Date=sku_city_ts["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "locality_intelligence": loc_intel.assign(Date=loc_intel["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
        
    print(f"    -> {excel_path.name}")
    print(f"    -> {json_path.name}")



# ── LAYER 11: SKU × CITY DRIVER ENRICHMENT ────────────────────────────────────

def enrich_sku_city_ts(full):
    """Adds 15 analysis columns to the sku_city_day master table:
    DoD for all drivers, zone status, consecutive signals,
    city rank, efficiency ratios, vs-national deltas."""
    print("[LLM Evidence] Enriching SKU x City with driver intelligence ...")

    full = full.sort_values(["Product ID", "City", "Date"]).copy()

    # Block 1 — DoD for OSA, Discount, Ad_SOV
    for col in ["OSA", "Discount", "Ad_SOV"]:
        full[f"{col}_DoD"] = full.groupby(["Product ID", "City"])[col].diff().fillna(0)

    # Block 2 — Zone Status
    def osa_zone(v):
        if v < THR["OSA_critical"]:  return "CRITICAL"
        if v < THR["OSA_warning"]:   return "WARNING"
        return "SAFE"

    def disc_zone(v):
        return "HIGH" if v > THR["Discount_dimret"] else "OK"

    def sov_zone(v):
        if v > THR["SOV_saturation"]: return "SATURATED"
        if v > 10:                    return "MODERATE"
        return "LOW"

    full["OSA_Zone"]      = full["OSA"].apply(osa_zone)
    full["Discount_Zone"] = full["Discount"].apply(disc_zone)
    full["SOV_Zone"]      = full["Ad_SOV"].apply(sov_zone)

    # Block 3 — Consecutive Decline / Rise per (Product ID, City)
    def consecutive_signal(series, direction="decline"):
        """Count consecutive days of decline (<0) or rise (>0)."""
        result = []
        count = 0
        for v in series:
            if direction == "decline" and v < 0:
                count += 1
            elif direction == "rise" and v > 0:
                count += 1
            else:
                count = 0
            result.append(count)
        return result

    for (pid, city), grp in full.groupby(["Product ID", "City"]):
        idx = grp.index
        full.loc[idx, "OSA_Consecutive_Decline_Days"]      = consecutive_signal(grp["OSA_DoD"].values, "decline")
        full.loc[idx, "Discount_Consecutive_Rise_Days"]    = consecutive_signal(grp["Discount_DoD"].values, "rise")
        full.loc[idx, "SOV_Consecutive_Decline_Days"]      = consecutive_signal(grp["Ad_SOV_DoD"].values, "decline")

    # Block 4 — City Rank within SKU per day (Rank 1 = worst)
    full["OSA_City_Rank"]      = full.groupby(["Date", "Product ID"])["OSA"].rank(ascending=True).fillna(0).astype(int)
    full["Discount_City_Rank"] = full.groupby(["Date", "Product ID"])["Discount"].rank(ascending=False).fillna(0).astype(int)

    # Block 5 — vs National SKU Average (how far above/below national mean)
    nat_avg = full.groupby(["Date", "Product ID"])[["OSA", "Discount"]].transform("mean")
    full["OSA_vs_National"]      = (full["OSA"]      - nat_avg["OSA"]).round(2)
    full["Discount_vs_National"] = (full["Discount"] - nat_avg["Discount"]).round(2)

    # Block 6 — Efficiency Ratios
    full["OSA_Efficiency"]  = (full["Revenue"] / full["OSA"].replace(0, float("nan"))).round(0)
    full["SOV_Efficiency"]  = (full["Revenue"] / full["Ad_SOV"].replace(0, float("nan"))).round(0)

    # Discount elasticity — only where Discount moved
    dod_disc = full["Discount_DoD"].replace(0, float("nan"))
    full["Discount_Elasticity"] = (full["Revenue_DoD"] / dod_disc).round(2)

    print(f"    -> Enriched {len(full):,} rows with 15 new driver intelligence columns")
    return full


# ── LAYER 12: SKU × CITY CAUSAL ATTRIBUTION ───────────────────────────────────

def _fit_sku_city_regression(history_df, drivers, nat_lag=None):
    """Fit a multivariate OLS regression for drivers vs Revenue per SKU x City."""
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy import stats as sp_stats
    import warnings
    
    sens = {}
    best_lags = {}
    
    # 1. Determine best univariate lag (0-3 days) for each driver
    for d in drivers:
        best_r = 0.0
        best_lag = 0
        for lag in range(4):
            sh = history_df[d].shift(lag)
            mask = sh.notna() & history_df["Revenue"].notna()
            if mask.sum() >= 5:
                if len(sh[mask].unique()) > 1:
                    r, _ = sp_stats.pearsonr(sh[mask], history_df["Revenue"][mask])
                    if not np.isnan(r) and abs(r) > abs(best_r):
                        best_r = r
                        best_lag = lag
        best_lags[d] = best_lag

    # 2. Build aligned multivariate dataframe
    ols_df = history_df[["Revenue"]].copy()
    for d in drivers:
        ols_df[d] = history_df[d].shift(best_lags[d])
    
    ols_df = ols_df.dropna()
    
    # 3. Fit Multivariate Regression if enough points
    n = len(ols_df)
    p = len(drivers)
    if n >= p + 2:
        X = ols_df[drivers].values
        y = ols_df["Revenue"].values
        
        reg = LinearRegression().fit(X, y)
        coefs = reg.coef_
        r2 = reg.score(X, y)
        
        # Calculate p-values manually
        dof = n - p - 1
        predictions = reg.predict(X)
        mse = np.sum((y - predictions)**2) / (dof if dof > 0 else 1)
        
        X_design = np.hstack([np.ones((n, 1)), X])
        try:
            var_b = mse * np.linalg.inv(np.dot(X_design.T, X_design)).diagonal()
            se_b = np.sqrt(var_b)[1:]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat = coefs / (se_b + 1e-9)
            p_values = [2 * (1 - sp_stats.t.cdf(np.abs(t), dof)) for t in t_stat]
        except np.linalg.LinAlgError:
            p_values = [1.0] * p
            
        for i, d in enumerate(drivers):
            sens[d] = {"coef": coefs[i], "p": p_values[i], "r2": r2, "weight": abs(coefs[i])}
    else:
        # Fall back to univariate regression if not enough points for full multivariate
        for d in drivers:
            sens[d] = {"coef": 0.0, "p": 1.0, "r2": 0.0, "weight": 0.0}
            sh = history_df[d].shift(best_lags[d])
            mask = sh.notna() & history_df["Revenue"].notna()
            if mask.sum() >= 5:
                if len(sh[mask].unique()) > 1:
                    res = sp_stats.linregress(sh[mask], history_df["Revenue"][mask])
                    if not np.isnan(res.slope):
                        sens[d] = {"coef": res.slope, "p": res.pvalue, "r2": res.rvalue**2, "weight": abs(res.slope)}
                    
    return sens


def _get_historical_proof_city(history_df, driver_col, target_delta, impact_rs, d1_date, d2_date):
    """Find a past instance where this driver moved similarly for this SKU×City."""
    import numpy as np
    if abs(target_delta) < 0.01:
        return "No change in this driver."
    best_match = None
    best_diff = float("inf")
    dates = sorted(history_df["Date"].unique())
    for i in range(1, len(dates)):
        hd1, hd2 = dates[i-1], dates[i]
        if pd.Timestamp(hd1) == pd.Timestamp(d1_date) and pd.Timestamp(hd2) == pd.Timestamp(d2_date):
            continue
        row1 = history_df[history_df["Date"] == hd1]
        row2 = history_df[history_df["Date"] == hd2]
        if row1.empty or row2.empty:
            continue
        drv_delta = float(row2.iloc[0][driver_col]) - float(row1.iloc[0][driver_col])
        rev_delta = float(row2.iloc[0]["Revenue"]) - float(row1.iloc[0]["Revenue"])
        # Same direction and magnitude within 50%
        if target_delta * drv_delta > 0 and abs(drv_delta) > 0:
            ratio = abs(drv_delta - target_delta) / (abs(target_delta) + 1e-9)
            if ratio < best_diff:
                best_diff = ratio
                best_match = {"d1": hd1.strftime("%b %d"), "d2": hd2.strftime("%b %d"),
                              "drv_delta": drv_delta, "rev_delta": rev_delta}
    if best_match and best_diff < 0.75:
        dir_w = "dropped" if target_delta < 0 else "rose"
        r_dir = "dropped" if best_match["rev_delta"] < 0 else "rose"
        return (f"{best_match['d1']}→{best_match['d2']}: {driver_col} {dir_w} "
                f"{abs(best_match['drv_delta']):.2f} → Revenue {r_dir} Rs.{abs(best_match['rev_delta']):,.0f}")
    return "No direct past match found."


def _build_attribution_row(grp_ts, d1_vals, d2_vals, drivers, sens, rev_delta, d1_date, d2_date):
    """Build one attribution dict using Regression coefficients."""
    import numpy as np
    
    MATERIALITY_THRESHOLDS = {
        "OSA": 1.0,
        "Discount": 3.0,
        "Ad_SOV": 1.0,
        "Network_Strength": 0.5
    }
    
    attr = {}
    max_abs = 0
    primary_driver = "Organic/Market"
    unexplained_impact = rev_delta
    
    for d in drivers:
        curr = d2_vals.get(d, 0)
        base = d1_vals.get(d, 0)
        dd   = curr - base
        
        coef = sens[d]["coef"]
        raw_impact = dd * coef
        
        mat_threshold = MATERIALITY_THRESHOLDS.get(d, 0.0)
        if abs(dd) < mat_threshold:
            impact = 0.0
            credibility = "MATERIALITY_FAILED"
            cred_str = f"Change of {dd:.2f} is below materiality threshold ({mat_threshold})."
        elif sens[d]["p"] > 0.10:
            impact = 0.0
            credibility = "STATISTICAL_FAILED"
            cred_str = f"Not statistically significant (p={sens[d]['p']:.3f} > 0.10)."
        else:
            impact = raw_impact
            credibility = "STRONG" if sens[d]["p"] < 0.05 else "MEDIUM"
            cred_str = f"Regression model: coef={coef:.1f} (p={sens[d]['p']:.3f}, R2={sens[d]['r2']:.2f})."
            
        unexplained_impact -= impact
        
        if impact != 0:
            proof = (f"Multivariate OLS over 20 days controls for confounders. "
                     f"Every 1 unit change in {d} explains Rs. {coef:,.0f} impact independently. "
                     f"Today's {dd:.2f} delta * {coef:,.0f} = Rs. {impact:,.0f}")
        else:
            proof = "N/A - Impact nullified by gates."
        
        attr[d] = {
            "d1": round(base, 2), "d2": round(curr, 2),
            "delta": round(dd, 2), "impact_rs": round(impact, 0),
            "weight": round(coef, 3), "r": round(sens[d].get("r", 0), 2),
            "p": round(sens[d]["p"], 3), "r2": round(sens[d]["r2"], 2),
            "confidence": credibility, "credibility_sentence": cred_str,
            "proof": proof
        }
        
        if abs(impact) > max_abs and impact != 0:
            max_abs = abs(impact)
            primary_driver = d
            
    return attr, primary_driver, unexplained_impact


def _network_quality_narrative(d1_row, d2_row):
    """Build a narrative for darkstore quality change."""
    if d1_row is None or d2_row is None:
        return ""
    dc1 = d1_row.get("Darkstore_Count", 0)
    dc2 = d2_row.get("Darkstore_Count", 0)
    hv1 = d1_row.get("High_Value_Store_Count", 0)
    hv2 = d2_row.get("High_Value_Store_Count", 0)
    ns1 = d1_row.get("Network_Strength", 0)
    ns2 = d2_row.get("Network_Strength", 0)
    dc_delta = dc2 - dc1
    hv_delta = hv2 - hv1
    lv_delta = (dc2 - hv2) - (dc1 - hv1)

    if dc_delta == 0:
        return f"Store count unchanged ({int(dc2)} stores). Network strength: {ns1:.1f}→{ns2:.1f}."

    action = f"Lost {abs(dc_delta):.0f}" if dc_delta < 0 else f"Gained {dc_delta:.0f}"
    quality = ""
    if hv_delta < 0:
        quality = f" ({abs(hv_delta):.0f} were HIGH-VALUE, locality score ≥0.00168)"
    elif hv_delta > 0:
        quality = f" ({hv_delta:.0f} were HIGH-VALUE)"
    cap = f"Network strength {ns1:.1f}→{ns2:.1f}."
    return f"{action} stores{quality}. {cap}"


def _get_competitor_snapshot(df_all, category, city, d1_date, d2_date):
    """Return competitor SOV, Discount, Revenue for this category×city on d1 and d2."""
    if not category or category == "Unknown":
        return {}
    comp = df_all[
        (df_all["Category"] == category) &
        (df_all["city_key"] == city.lower().strip()) &
        (df_all["Brand"] != BRAND_RAW)
    ]
    result = {}
    for label, dt in [("d1", pd.Timestamp(d1_date)), ("d2", pd.Timestamp(d2_date))]:
        day = comp[comp["Date"] == dt]
        if day.empty:
            result[label] = {}
            continue
        by_brand = day.groupby("Brand").agg(
            Rev=("Offtake MRP", "sum"),
            SOV=("Ad SOV", "mean"),
            Disc=("Wt. Discount %", "mean")
        )
        result[label] = {
            "total_comp_rev": round(by_brand["Rev"].sum(), 0),
            "avg_comp_sov":   round(by_brand["SOV"].mean(), 2),
            "avg_comp_disc":  round(by_brand["Disc"].mean(), 2),
            "top_brand_by_rev": by_brand["Rev"].idxmax() if not by_brand.empty else "N/A"
        }

    # Aggressive mover
    aggressive = "None"
    if result.get("d1") and result.get("d2"):
        d1_day = comp[comp["Date"] == pd.Timestamp(d1_date)]
        d2_day = comp[comp["Date"] == pd.Timestamp(d2_date)]
        if not d1_day.empty and not d2_day.empty:
            b1 = d1_day.groupby("Brand").agg(SOV=("Ad SOV", "mean"), Disc=("Wt. Discount %", "mean"))
            b2 = d2_day.groupby("Brand").agg(SOV=("Ad SOV", "mean"), Disc=("Wt. Discount %", "mean"))
            diff = b2.sub(b1, fill_value=0)
            if not diff.empty and diff["SOV"].max() > 1.0:
                top = diff["SOV"].idxmax()
                aggressive = f"{top.title()} (+{diff['SOV'].max():.1f}% SOV)"
    result["aggressive_mover"] = aggressive
    return result


def _detect_combinations(d2_row, comp):
    """Detect dangerous driver combinations and return list of active flags."""
    flags = []
    zone_osa   = d2_row.get("OSA_Zone", "SAFE")
    zone_disc  = d2_row.get("Discount_Zone", "OK")
    zone_sov   = d2_row.get("SOV_Zone", "LOW")
    hv_pct     = d2_row.get("High_Value_Store_Pct", 100)
    consec_osa = d2_row.get("OSA_Consecutive_Decline_Days", 0)

    comp_sov_rising = False
    comp_disc_rising = False
    if comp.get("d1") and comp.get("d2"):
        comp_sov_rising  = comp["d2"].get("avg_comp_sov", 0)  > comp["d1"].get("avg_comp_sov", 0)
        comp_disc_rising = comp["d2"].get("avg_comp_disc", 0) > comp["d1"].get("avg_comp_disc", 0)

    if zone_osa in ("WARNING", "CRITICAL") and zone_sov in ("MODERATE", "SATURATED"):
        flags.append("AD_WASTAGE: Ads running but OSA below safe zone — customers can't buy")
    if zone_osa == "CRITICAL" and comp_disc_rising:
        flags.append("DOUBLE_LOSS: OSA critical + competitor increasing discount")
    if zone_osa == "CRITICAL" and zone_disc == "OK":
        flags.append("OSA_ONLY: Availability collapse — no discount buffer")
    if hv_pct < 30 and consec_osa >= 2:
        flags.append("NETWORK_QUALITY_CRISIS: Low high-value store % + OSA declining")
    if comp_sov_rising and zone_sov == "LOW":
        flags.append("VISIBILITY_GAP: Competitor increasing ads while our SOV is low")
    if zone_disc == "HIGH":
        flags.append("DISCOUNT_HIGH: Discount above 20% — diminishing returns risk")
    if not flags:
        flags.append("NO_ACTIVE_RISK_COMBINATION")
    return flags


def build_sku_city_attribution(sku_city_ts, df24, df_all, d1_str, d2_str, mode="day", nat_lag=None):
    """
    Build Rs. attribution for every (SKU, City) pair.
    mode='day':    compare single d1_str vs d2_str
    mode='period': compare avg of period1 vs avg of period2 (d1_str=start1, d2_str=end1 not used for period)
    """
    label = f"[LLM Evidence] Building SKU x City attribution ({mode}) ..."
    print(label)

    drivers = ["OSA", "Discount", "Ad_SOV", "Network_Strength", "Organic_SOV"]
    rows = []

    grouped = sku_city_ts.groupby(["Product ID", "Product Name", "Grammage", "City"])

    for (pid, pname, grammage, city), grp in grouped:
        grp = grp.sort_values("Date").copy()

        # ─── Snapshot extraction ─────────────────────────────────────
        if mode == "day":
            d1 = pd.Timestamp(d1_str)
            d2 = pd.Timestamp(d2_str)
            r1 = grp[grp["Date"] == d1]
            r2 = grp[grp["Date"] == d2]
            if r1.empty or r2.empty:
                continue
            s1 = r1.iloc[0].to_dict()
            s2 = r2.iloc[0].to_dict()
        else:  # period mode
            p1s, p1e = pd.Timestamp(COMPARE_PERIOD1_START), pd.Timestamp(COMPARE_PERIOD1_END)
            p2s, p2e = pd.Timestamp(COMPARE_PERIOD2_START), pd.Timestamp(COMPARE_PERIOD2_END)
            p1 = grp[(grp["Date"] >= p1s) & (grp["Date"] <= p1e)]
            p2 = grp[(grp["Date"] >= p2s) & (grp["Date"] <= p2e)]
            if p1.empty or p2.empty:
                continue
            
            num_cols = drivers + ["Revenue", "Darkstore_Count", "High_Value_Store_Count", "Network_Strength", "High_Value_Store_Pct", "OSA_Consecutive_Decline_Days"]
            str_cols = ["OSA_Zone", "Discount_Zone", "SOV_Zone"]
            
            s2_num = p1[num_cols].mean().to_dict()
            s2_str = p1[str_cols].iloc[-1].to_dict()
            s2 = {**s2_num, **s2_str}
            
            s1_num = p2[num_cols].mean().to_dict()
            s1_str = p2[str_cols].iloc[-1].to_dict()
            s1 = {**s1_num, **s1_str}
            
            d1 = p2s
            d2 = p1s

        rev_delta = s2.get("Revenue", 0) - s1.get("Revenue", 0)

        # ─── Correlations per (SKU, City) history ────────────────────
        sens = _fit_sku_city_regression(grp, drivers, nat_lag)

        # ─── Driver attribution ───────────────────────────────────────
        d1_vals = {d: s1.get(d, 0) for d in drivers}
        d2_vals = {d: s2.get(d, 0) for d in drivers}
        attr, primary, unexplained = _build_attribution_row(grp, d1_vals, d2_vals, drivers, sens, rev_delta, d1, d2)

        # ─── Darkstore quality narrative ──────────────────────────────
        net_narrative = _network_quality_narrative(s1, s2)

        # ─── Competitor snapshot ──────────────────────────────────────
        category = df24[(df24["Product ID"] == pid)]["Category"].iloc[0] \
                   if not df24[(df24["Product ID"] == pid)].empty else "Unknown"
        city_key = city.lower().strip()
        comp = _get_competitor_snapshot(df_all, category, city_key, d1, d2)

        # ─── Combination flags ────────────────────────────────────────
        combos = _detect_combinations(s2, comp)

        # ─── Build flat output row ────────────────────────────────────
        row = {
            "Product ID": pid, "Product Name": pname, "Grammage": grammage, "City": city,
            "Category": category,
            "Revenue_d1": round(s1.get("Revenue", 0), 0),
            "Revenue_d2": round(s2.get("Revenue", 0), 0),
            "Revenue_Delta": round(rev_delta, 0),
            "Primary_Driver": primary,
            "Unexplained_Organic_Impact_Rs": round(unexplained, 0),

            # Per-driver attribution
            "OSA_d1": attr["OSA"]["d1"], "OSA_d2": attr["OSA"]["d2"],
            "OSA_Delta": attr["OSA"]["delta"], "OSA_Impact_Rs": attr["OSA"]["impact_rs"],
            "OSA_Confidence": attr["OSA"]["confidence"], "OSA_Credibility": attr["OSA"]["credibility_sentence"],
            "OSA_Proof": attr["OSA"]["proof"],

            "Discount_d1": attr["Discount"]["d1"], "Discount_d2": attr["Discount"]["d2"],
            "Discount_Delta": attr["Discount"]["delta"], "Discount_Impact_Rs": attr["Discount"]["impact_rs"],
            "Discount_Confidence": attr["Discount"]["confidence"], "Discount_Credibility": attr["Discount"]["credibility_sentence"],
            "Discount_Proof": attr["Discount"]["proof"],

            "Ad_SOV_d1": attr["Ad_SOV"]["d1"], "Ad_SOV_d2": attr["Ad_SOV"]["d2"],
            "Ad_SOV_Delta": attr["Ad_SOV"]["delta"], "Ad_SOV_Impact_Rs": attr["Ad_SOV"]["impact_rs"],
            "Ad_SOV_Confidence": attr["Ad_SOV"]["confidence"], "Ad_SOV_Credibility": attr["Ad_SOV"]["credibility_sentence"],
            "Ad_SOV_Proof": attr["Ad_SOV"]["proof"],

            "Network_d1": attr["Network_Strength"]["d1"], "Network_d2": attr["Network_Strength"]["d2"],
            "Network_Delta": attr["Network_Strength"]["delta"],
            "Network_Impact_Rs": attr["Network_Strength"]["impact_rs"],
            "Network_Confidence": attr["Network_Strength"]["confidence"], "Network_Credibility": attr["Network_Strength"]["credibility_sentence"],
            "Network_Proof": attr["Network_Strength"]["proof"],
            "Network_Quality_Narrative": net_narrative,

            # Zone status on d2
            "OSA_Zone_d2": s2.get("OSA_Zone", ""),
            "Discount_Zone_d2": s2.get("Discount_Zone", ""),
            "SOV_Zone_d2": s2.get("SOV_Zone", ""),
            "OSA_Consecutive_Decline_Days": s2.get("OSA_Consecutive_Decline_Days", 0),

            # Competitor
            "Comp_Rev_d1": comp.get("d1", {}).get("total_comp_rev", "N/A"),
            "Comp_Rev_d2": comp.get("d2", {}).get("total_comp_rev", "N/A"),
            "Comp_Avg_SOV_d1": comp.get("d1", {}).get("avg_comp_sov", "N/A"),
            "Comp_Avg_SOV_d2": comp.get("d2", {}).get("avg_comp_sov", "N/A"),
            "Comp_Avg_Disc_d1": comp.get("d1", {}).get("avg_comp_disc", "N/A"),
            "Comp_Avg_Disc_d2": comp.get("d2", {}).get("avg_comp_disc", "N/A"),
            "Aggressive_Competitor": comp.get("aggressive_mover", "None"),

            # Combination flags
            "Active_Combinations": " | ".join(combos),
        }
        rows.append(row)

    df_attr = pd.DataFrame(rows)
    if not df_attr.empty:
        df_attr = df_attr.sort_values("Revenue_Delta")
    print(f"    -> {len(df_attr)} SKU x City attribution rows built")
    return df_attr


if __name__ == "__main__":
    main()
