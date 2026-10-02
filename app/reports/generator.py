"""Automated report generation (self-contained HTML with interactive Plotly charts, or Markdown)."""
from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from app.ai.schemas import Narrative
from app.core import anomaly, trends, visuals
from app.core.insights import _fmt

CSS = """
body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#f5f6fa;color:#1f2937}
.wrap{max-width:1050px;margin:0 auto;padding:32px 20px}
h1{margin:0 0 4px}h2{margin-top:36px;border-bottom:2px solid #e5e7eb;padding-bottom:6px}
.sub{color:#6b7280;margin-bottom:24px}
.card{background:#fff;border-radius:12px;padding:18px 22px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin:14px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:#fff;border-radius:12px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.kpi .v{font-size:24px;font-weight:700}.kpi .n{color:#6b7280;font-size:13px}
.up{color:#059669}.down{color:#dc2626}
.sev{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600;margin-right:8px}
.critical{background:#fee2e2;color:#991b1b}.warning{background:#fef3c7;color:#92400e}.info{background:#dbeafe;color:#1e40af}
details{margin-top:8px;color:#4b5563;font-size:13px}pre{background:#f3f4f6;padding:10px;border-radius:8px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #e5e7eb;padding:6px 8px;text-align:left}
"""


def _fmt_kpi(k: Dict[str, Any]) -> str:
    if k.get("format") == "pct":
        return f"{k['value']:.1f}%"
    return _fmt(k["value"]) if k["format"] != "int" else f"{int(k['value']):,}"


def _executive_summary(res: Dict[str, Any], llm) -> Dict[str, Any]:
    ins = res["insights"]
    top = [i for i in ins if i["severity"] in ("critical", "warning")][:4] or ins[:3]
    if llm:
        try:
            n = llm.structured(Narrative, "You are a BI analyst. Using ONLY the supplied verified insights, write an executive summary "
                               "(≤120 words), likely root causes, and prioritised recommendations. Do not invent numbers.",
                               str([{k: i[k] for k in ("title", "severity", "description", "recommendation")} for i in ins[:12]]))
            return n.model_dump()
        except Exception:
            pass
    summary = " ".join(i["description"] for i in top[:3]) if top else "No material issues were detected."
    causes = [i["description"] for i in ins if i["category"] == "driver"][:2]
    recs = list(dict.fromkeys(i["recommendation"] for i in ins if i["recommendation"]))[:6]
    return {"executive_summary": summary, "root_causes": causes, "recommendations": recs}


def _figures(res: Dict[str, Any], df: pd.DataFrame) -> List[Any]:
    figs = []
    dcol, metric = res.get("date_column"), res.get("primary_metric")
    prof = res["profile"]
    if dcol and metric:
        ser = trends.aggregate_series(df, dcol, metric, "sum")
        figs.append(("Trend", visuals.time_series(ser, f"{metric} by period")))
        daily = trends.aggregate_series(df, dcol, metric, "sum", "D")
        ta = anomaly.time_series_anomalies(daily)
        figs.append(("Daily anomalies", visuals.time_series(daily, f"Daily {metric} (anomalies marked)", ta)))
    drv = res.get("drivers")
    if drv and drv["drivers"]:
        figs.append(("Change drivers", visuals.waterfall(drv["drivers"][:5], drv["previous_total"], drv["current_total"])))
    cats = [c["name"] for c in prof["column_profiles"] if c["semantic_type"] == "categorical"]
    if cats and metric:
        figs.append((f"{metric} by {cats[0]}", visuals.bar_by_category(df, cats[0], metric)))
    heat = visuals.correlation_heatmap(df, [c for c in prof["numeric_columns"] if not c.endswith("_id")][:10])
    if heat is not None:
        figs.append(("Correlations", heat))
    return figs


def build_report(res: Dict[str, Any], df: pd.DataFrame, title: str = "Analytics Report", fmt: str = "html", llm=None) -> str:
    nar = _executive_summary(res, llm)
    prof = res["profile"]
    figs = _figures(res, df)
    if fmt == "md":
        L = [f"# {title}", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_ · dataset `{prof['name']}` · {prof['rows']:,} rows × {prof['columns']} columns · quality {prof['quality_score']}/100", "",
             "## Executive summary", nar["executive_summary"], "", "## KPIs"]
        L += [f"- **{k['name']}**: {_fmt_kpi(k)}" + (f" ({k['change_pct']:+.1f}% vs previous period)" if k.get("change_pct") is not None else "") for k in res["kpis"]]
        L += ["", "## Key findings"]
        for i in res["insights"]:
            L += [f"### [{i['severity'].upper()}] {i['title']}", i["description"], f"_Recommendation:_ {i['recommendation']}", ""]
        L += ["## Root causes"] + [f"- {c}" for c in nar["root_causes"]] + ["", "## Recommendations"] + [f"{n}. {r}" for n, r in enumerate(nar["recommendations"], 1)]
        L += ["", "## Data cleaning log"] + [f"- {a['action']} {a['column'] or ''}: {a['count']} {a['detail']}" for a in res["cleaning_log"]]
        return "\n".join(L)

    e = html.escape
    kp = "".join(
        f"<div class='kpi'><div class='n'>{e(k['name'])}</div><div class='v'>{e(_fmt_kpi(k))}</div>"
        + (f"<div class='{'up' if k['change_pct'] >= 0 else 'down'}'>{k['change_pct']:+.1f}% vs prev. period</div>" if k.get("change_pct") is not None else "")
        + "</div>" for k in res["kpis"])
    ins = "".join(
        f"<div class='card'><span class='sev {i['severity']}'>{i['severity']}</span><b>{e(i['title'])}</b><p>{e(i['description'])}</p>"
        f"<p><b>Recommendation:</b> {e(i['recommendation'])}</p>"
        f"<details><summary>Evidence</summary><pre>{e(str({k: v for k, v in i['evidence'].items() if k not in ('drill_down',)})[:1500])}</pre></details></div>"
        for i in res["insights"])
    figs_html = "".join(f"<div class='card'>{f.to_html(full_html=False, include_plotlyjs=False)}</div>" for _, f in figs)
    clog = "".join(f"<tr><td>{e(a['action'])}</td><td>{e(str(a['column'] or '—'))}</td><td>{a['count']}</td><td>{e(a['detail'])}</td></tr>" for a in res["cleaning_log"])
    ul = lambda xs: "<ul>" + "".join(f"<li>{e(x)}</li>" for x in xs) + "</ul>" if xs else "<p>—</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>{CSS}</style></head><body><div class="wrap">
<h1>{e(title)}</h1><div class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} · dataset <b>{e(prof['name'])}</b> · {prof['rows']:,} rows × {prof['columns']} columns · data quality {prof['quality_score']}/100</div>
<h2>Executive summary</h2><div class="card">{e(nar['executive_summary'])}</div>
<h2>KPIs</h2><div class="kpis">{kp}</div>
<h2>Key findings</h2>{ins}
<h2>Root causes</h2><div class="card">{ul(nar['root_causes'])}</div>
<h2>Recommendations</h2><div class="card">{ul(nar['recommendations'])}</div>
<h2>Visual analysis</h2>{figs_html}
<h2>Data cleaning log</h2><div class="card"><table><tr><th>Action</th><th>Column</th><th>Count</th><th>Detail</th></tr>{clog}</table></div>
<p class="sub">All figures are computed with Pandas/SciPy/scikit-learn from the loaded data; evidence for every finding is listed under each item.</p>
</div></body></html>"""
