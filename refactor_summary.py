import re

file_path = r'C:\Users\cpsge\.gemini\antigravity\scratch\Project - Sales RCA Reasoning\advanced_rca_engine.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

new_sku_deep_dive = '''
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
            
            if total_drop < 0 and impact > 0: continue
            if total_drop > 0 and impact < 0: continue
            
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
        
        if not (total_drop < 0 and unexplained > 0) and not (total_drop > 0 and unexplained < 0):
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
        if total_drop < 0 and imp > 0: continue
        if total_drop > 0 and imp < 0: continue
        
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
'''

pattern = re.compile(r'def sku_deep_dive\(df24, df_all, df1, sku_id, sku_name, d1_str, d2_str\):.*?return \{"sku_name": sku_name.*?\}', re.DOTALL)
content = pattern.sub(new_sku_deep_dive.strip(), content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("SKU Deep Dive replaced with Two-Pass Logic.")
