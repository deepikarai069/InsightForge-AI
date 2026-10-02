"""Autonomous analysis orchestrator: profile → EDA → stats → trends → anomalies → evidence-backed insights."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import anomaly, kpi, stats as st, trends
from .profiling import profile_dataframe


@dataclass
class Insight:
    title: str
    category: str            # data_quality | trend | anomaly | correlation | segment | driver | kpi
    severity: str            # info | warning | critical
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:.2f}B"
    if a >= 1e6:
        return f"{v/1e6:.2f}M"
    if a >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:.2f}" if a < 100 and v != int(v) else f"{v:,.0f}"


def pick_date_column(profile: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
    cands = profile["datetime_columns"]
    if not cands:
        return None
    strong = ("order", "transaction", "event", "created", "timestamp", "sale", "purchase", "invoice", "booking")
    pos = {c: i for i, c in enumerate(df.columns)}
    pri = sorted(cands, key=lambda c: (0 if any(h in c for h in strong) else 1, pos[c]))
    return pri[0]


def pick_primary_metric(profile: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
    for hint in ("revenue", "sales", "amount", "profit", "total", "value", "price", "quantity"):
        for c in profile["numeric_columns"]:
            if hint in c.lower():
                return c
    return profile["numeric_columns"][0] if profile["numeric_columns"] else None


def run_auto_analysis(df: pd.DataFrame, name: str = "dataset", cleaning_log: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Run the full analytic pipeline on one (already joined, if needed) DataFrame."""
    prof = profile_dataframe(df, name)
    date_col = pick_date_column(prof, df)
    metric = pick_primary_metric(prof, df)
    num = prof["numeric_columns"]
    cat = [c["name"] for c in prof["column_profiles"] if c["semantic_type"] == "categorical"]  # dimensions (no binary flags)
    insights: List[Insight] = []

    # --- data quality -----------------------------------------------------
    high_missing = [c for c in prof["column_profiles"] if c["missing_pct"] >= 5]
    if high_missing:
        worst = max(high_missing, key=lambda c: c["missing_pct"])
        insights.append(Insight(
            "Missing data detected", "data_quality", "warning" if worst["missing_pct"] < 30 else "critical",
            f"{len(high_missing)} column(s) have ≥5% missing values; worst is '{worst['name']}' at {worst['missing_pct']}%.",
            {"columns": {c["name"]: c["missing_pct"] for c in high_missing}},
            "Fix upstream collection for these fields or confirm imputation assumptions before relying on them."))
    if cleaning_log:
        dup = sum(a["count"] for a in cleaning_log if a["action"] == "drop_duplicate_rows")
        if dup:
            insights.append(Insight("Duplicate rows removed", "data_quality", "info",
                                    f"{dup} exact duplicate rows were removed during cleaning.",
                                    {"duplicates_removed": dup},
                                    "Check the source export/ETL for double-loading."))

    # --- KPIs -------------------------------------------------------------
    kpis = kpi.compute_kpis(df, date_col)

    # --- trends -------------------------------------------------------------
    series_cache: Dict[str, pd.Series] = {}
    series_agg: Dict[str, str] = {}
    if date_col:
        for k in kpis:
            col = k.get("column")
            if not col or k.get("format") == "pct" or col in series_cache:
                continue
            s = trends.aggregate_series(df, date_col, col, k.get("agg", "sum"))
            series_cache[col] = s
            series_agg[col] = k.get("agg", "sum")
            tr = trends.detect_trend(s)
            if tr["direction"] in {"upward", "downward"}:
                up = tr["direction"] == "upward"
                insights.append(Insight(
                    f"{col.replace('_', ' ').title()} is trending {tr['direction']}", "trend",
                    "info" if up else "warning",
                    f"{col} moved from {_fmt(tr['first_value'])} to {_fmt(tr['last_value'])} "
                    f"({tr['pct_change_first_last']:+.1f}%) between {tr['start']} and {tr['end']}; "
                    f"linear trend {tr['relative_slope_pct']:+.2f}% of mean per period (R²={tr['r_squared']}, p={tr['p_value']:.3g}).",
                    tr, "Sustain the drivers of growth." if up else "Investigate the drivers of decline (see root-cause section)."))
            ch = trends.latest_period_change(s)
            if ch and abs(ch["pct_change"]) >= 10:
                insights.append(Insight(
                    f"{col.replace('_', ' ').title()} {'jumped' if ch['pct_change'] > 0 else 'fell'} {abs(ch['pct_change']):.1f}% in the latest period",
                    "trend", "warning" if ch["pct_change"] < 0 else "info",
                    f"{ch['previous_period']}: {_fmt(ch['previous'])} → {ch['current_period']}: {_fmt(ch['current'])}.",
                    ch, "Run a driver analysis on this metric." if ch["pct_change"] < 0 else "Identify what worked and replicate it."))

    # --- root cause on the primary metric ---------------------------------
    drivers = None
    if date_col and metric and cat:
        drv = trends.driver_analysis(df, date_col, metric, cat[:8])
        if "error" not in drv and drv.get("pct_change") is not None and abs(drv["pct_change"]) >= 5 and drv["drivers"]:
            drivers = drv
            top = drv["drivers"][0]
            txt = (f"{metric} changed {drv['pct_change']:+.1f}% ({_fmt(drv['delta'])}) between "
                   f"{drv['previous_period'][0]}…{drv['previous_period'][1]} and {drv['current_period'][0]}…{drv['current_period'][1]}. "
                   f"Largest contributor: {top['dimension']} = '{top['member']}' ({_fmt(top['delta'])}, {top['share_of_change_pct']}% of the change).")
            hs = drv.get("hotspot")
            if hs:
                cell = " & ".join(f"{k}='{v}'" for k, v in hs["cell"].items())
                txt += (f" Hotspot: {cell} accounts for {hs['share_of_change_pct']}% of the change while representing only "
                        f"{hs['base_share_pct']}% of the prior period ({hs['lift']}× concentration).")
            insights.append(Insight(f"Root cause: what moved {metric}", "driver",
                                    "critical" if drv["delta"] < 0 and abs(drv["pct_change"]) >= 15 else "warning" if drv["delta"] < 0 else "info",
                                    txt, drv,
                                    (f"Investigate {' & '.join(f'{k}={v}' for k, v in hs['cell'].items())} first — check outages, pricing, stock, campaigns or tracking changes there."
                                     if hs else f"Prioritise investigation of {top['dimension']}='{top['member']}'.")))

    # --- anomalies --------------------------------------------------------
    anomalies: Dict[str, Any] = {"univariate": [], "time_series": {}, "multivariate_rows": 0}
    ua = anomaly.univariate_anomalies(df, [c for c in num if not c.endswith("_id")])
    for f in ua["findings"][:3]:
        anomalies["univariate"].append({k: v for k, v in f.items() if k != "examples"})
        insights.append(Insight(f"Outliers in '{f['column']}'", "anomaly", "warning" if f["pct_of_rows"] > 1 else "info",
                                f"{f['count']} rows ({f['pct_of_rows']}%) deviate more than {f['threshold']} robust σ from the median.",
                                {k: v for k, v in f.items() if k != "index"},
                                "Validate these records — they may be data-entry errors or genuine high-value events."))
    if date_col:
        for col in [c for c in series_cache if series_agg[c] == "sum"][:3]:
            daily = trends.aggregate_series(df, date_col, col, "sum", "D")
            ta = anomaly.time_series_anomalies(daily)
            if len(ta):
                anomalies["time_series"][col] = ta.reset_index().rename(columns={"index": "date", date_col: "date"}).astype({"date": str}).head(10).to_dict(orient="records")
                w = ta.iloc[0]
                insights.append(Insight(f"Daily {col.replace('_', ' ')} anomaly on {str(ta.index[0].date())}", "anomaly",
                                        "warning", f"{len(ta)} anomalous day(s). Most extreme: {str(ta.index[0].date())} was a {w['direction']} "
                                        f"({_fmt(w['value'])} vs expected {_fmt(w['expected'])}, {w['deviation_pct']:+.0f}%).",
                                        {"top": ta.head(5).reset_index().astype({ta.index.name or "index": str}).to_dict(orient="records")},
                                        "Cross-check with campaigns, outages, holidays or data-loading issues on those dates."))
    mv_cols = [c for c in num if not c.endswith("_id")][:6]
    mv = anomaly.multivariate_anomalies(df, mv_cols)
    anomalies["multivariate_rows"] = int(len(mv))

    # --- correlations & segments -----------------------------------------
    corr = st.correlations(df, [c for c in num if not c.endswith("_id")])
    for c in corr[:3]:
        insights.append(Insight(f"{c['strength'].title()} relationship: {c['x']} ↔ {c['y']}", "correlation", "info",
                                f"Pearson r={c['pearson_r']} (Spearman ρ={c['spearman_rho']}, n={c['n']}, p={c['p_value']:.2g}). Correlation is not causation.",
                                c, "Check whether the link is mechanical (one derived from the other) before acting on it."))
    seg = st.group_differences(df, [c for c in num if not c.endswith("_id")][:6], cat[:6])
    for g in seg[:3]:
        insights.append(Insight(f"{g['metric']} differs by {g['dimension']}", "segment", "info",
                                f"Highest: {g['highest']['group']} (mean {_fmt(g['highest']['mean'])}); lowest: {g['lowest']['group']} "
                                f"(mean {_fmt(g['lowest']['mean'])}). η²={g['effect_size_eta2']}, p={g['p_value']:.2g} ({g['test']}).",
                                g, f"Tailor strategy for '{g['lowest']['group']}' or scale what works for '{g['highest']['group']}'."))

    order = {"critical": 0, "warning": 1, "info": 2}
    insights.sort(key=lambda i: order[i.severity])
    return {
        "profile": prof, "kpis": kpis, "date_column": date_col, "primary_metric": metric,
        "insights": [i.to_dict() for i in insights], "correlations": corr, "segments": seg,
        "anomalies": anomalies, "drivers": drivers, "cleaning_log": cleaning_log or [],
    }
