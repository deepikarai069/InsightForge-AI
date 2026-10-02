"""Streamlit front-end.  Run:  streamlit run app/ui/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import warnings

import pandas as pd
import plotly.io as pio
import streamlit as st

from app.core import trends, visuals
from app.services import InsightForgeService

warnings.filterwarnings("ignore")
st.set_page_config(page_title="InsightForge AI", page_icon="🔮", layout="wide")


@st.cache_resource
def service() -> InsightForgeService:
    return InsightForgeService()


svc = service()
SEV = {"critical": "🔴", "warning": "🟠", "info": "🔵"}

st.title("🔮 InsightForge AI")
st.caption("Autonomous data analytics & BI copilot · "
           f"LLM: **{svc.llm.name if svc.llm else 'offline heuristic engine'}** · vector store: **{svc.kb.backend}** · DB: **{svc.store.engine.dialect.name}**")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📁 Data")
    files = st.file_uploader("Upload CSV / Excel (multiple related files allowed)", type=["csv", "tsv", "xlsx", "xls"], accept_multiple_files=True)
    clean = st.checkbox("Auto-clean on upload", True)
    if files and st.button("Ingest files", type="primary"):
        for f in files:
            try:
                for r in svc.ingest_file(f.name, f.getvalue(), clean):
                    st.success(f"{r['name']}: {r['rows']:,} rows · quality {r['quality_score']}")
            except Exception as e:
                st.error(f"{f.name}: {e}")
    sample = Path("sample_data")
    if sample.exists() and st.button("Load demo e-commerce data"):
        for f in ["orders.csv", "customers.csv", "products.csv"]:
            svc.ingest_file(f, (sample / f).read_bytes(), True)
        g = sample / "business_glossary.md"
        if g.exists():
            svc.add_glossary("Business glossary", g.read_text())
        st.rerun()
    st.divider()
    st.header("📖 Business glossary")
    gl = st.file_uploader("Upload glossary (.md/.txt)", type=["md", "txt"], key="gl")
    if gl and st.button("Index glossary"):
        st.success(f"Indexed {svc.add_glossary(gl.name, gl.getvalue().decode('utf-8', 'ignore'))} chunks")
    ds = svc.store.list_datasets()
    if ds:
        st.divider()
        st.subheader("Loaded datasets")
        for d in ds:
            st.write(f"• **{d['name']}** — {d['rows']:,} × {d['columns']}")
        for r in svc.store.relationships():
            st.caption(f"🔗 {r['left_table']}.{r['left_column']} → {r['right_table']}.{r['right_column']} ({r['cardinality']})")

if not ds:
    st.info("👈 Upload one or more datasets (or load the demo data) to begin.")
    st.stop()

fact = st.selectbox("Primary (fact) table", [d["name"] for d in ds], index=max(range(len(ds)), key=lambda i: ds[i]["rows"]))


@st.cache_data(show_spinner="Running autonomous analysis…")
def analysis(fact_name: str, sig: str):
    return svc.auto_analysis(fact_name)


sig = json.dumps([(d["name"], d["rows"], d["created_at"]) for d in ds])
res = analysis(fact, sig)
frame = svc.store.joined_frame(fact)

tab_over, tab_ins, tab_chat, tab_explore, tab_sql, tab_report = st.tabs(
    ["📊 Dashboard", "💡 Insights", "💬 Ask your data", "🔎 Explore & quality", "🧮 SQL", "📄 Report"])

with tab_over:
    cols = st.columns(min(len(res["kpis"]), 5))
    for i, k in enumerate(res["kpis"][:10]):
        with cols[i % len(cols)]:
            v = f"{k['value']:.1f}%" if k.get("format") == "pct" else f"{k['value']:,.0f}" if abs(k["value"]) >= 100 else f"{k['value']:,.2f}"
            st.metric(k["name"], v, f"{k['change_pct']:+.1f}%" if k.get("change_pct") is not None else None)
    dcol, metric = res["date_column"], res["primary_metric"]
    if dcol and metric:
        ser = trends.aggregate_series(frame, dcol, metric, "sum", "D")
        from app.core.anomaly import time_series_anomalies
        st.plotly_chart(visuals.time_series(ser, f"Daily {metric}", time_series_anomalies(ser)), width="stretch")
        drv = res.get("drivers")
        if drv and drv["drivers"]:
            st.plotly_chart(visuals.waterfall(drv["drivers"][:5], drv["previous_total"], drv["current_total"]), width="stretch")
    cats = [c["name"] for c in res["profile"]["column_profiles"] if c["semantic_type"] == "categorical"]
    c1, c2 = st.columns(2)
    if cats and metric:
        with c1:
            st.plotly_chart(visuals.bar_by_category(frame, cats[0], metric), width="stretch")
        if len(cats) > 1:
            with c2:
                st.plotly_chart(visuals.bar_by_category(frame, cats[1], metric), width="stretch")

with tab_ins:
    for i in res["insights"]:
        with st.expander(f"{SEV[i['severity']]} {i['title']}", expanded=i["severity"] == "critical"):
            st.write(i["description"])
            if i["recommendation"]:
                st.success(f"**Recommendation:** {i['recommendation']}")
            st.json({k: v for k, v in i["evidence"].items() if k != "drill_down"}, expanded=False)

with tab_chat:
    st.caption("Try: *top 5 products by revenue* · *average order value by channel* · *why did revenue drop in December?* · *any anomalies in revenue?*")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            a = m.get("answer")
            if a:
                if a.chart_json:
                    st.plotly_chart(pio.from_json(a.chart_json), width="stretch")
                if a.rows:
                    st.dataframe(pd.DataFrame(a.rows), width="stretch", height=min(300, 38 * (len(a.rows) + 1)))
                with st.expander("Evidence & reasoning"):
                    if a.sql:
                        st.code(a.sql, language="sql")
                    st.write("Steps:", " → ".join(a.steps))
                    st.write("Number verification:", a.verification)
                    if a.context:
                        st.write("Retrieved context:", [c["title"] for c in a.context])
    if q := st.chat_input("Ask a question about your data"):
        st.session_state.chat.append({"role": "user", "content": q})
        a = svc.ask(q, "streamlit")
        st.session_state.chat.append({"role": "assistant", "content": a.answer, "answer": a})
        st.rerun()

with tab_explore:
    prof = res["profile"]
    st.metric("Data quality score", f"{prof['quality_score']}/100")
    st.dataframe(pd.DataFrame(prof["column_profiles"]).drop(columns=["top_values"], errors="ignore"), width="stretch")
    if res["cleaning_log"]:
        st.subheader("Cleaning log")
        st.dataframe(pd.DataFrame(res["cleaning_log"]), width="stretch")
    num = prof["numeric_columns"]
    if num:
        col = st.selectbox("Distribution of", num)
        st.plotly_chart(visuals.histogram(frame, col), width="stretch")
        heat = visuals.correlation_heatmap(frame, [c for c in num if not c.endswith("_id")][:12])
        if heat is not None:
            st.plotly_chart(heat, width="stretch")

with tab_sql:
    q = st.text_area("Read-only SQL", f"SELECT * FROM {fact} LIMIT 20", height=120)
    if st.button("Run"):
        try:
            st.dataframe(svc.run_sql(q), width="stretch")
        except Exception as e:
            st.error(str(e))
    st.caption("Only single SELECT statements on loaded tables are permitted.")

with tab_report:
    fmt = st.radio("Format", ["html", "md"], horizontal=True)
    if st.button("Generate report", type="primary"):
        st.session_state.report = (svc.report(fact, fmt=fmt), fmt)
    if "report" in st.session_state:
        body, f = st.session_state.report
        st.download_button(f"⬇️ Download .{f}", body, file_name=f"insightforge_report.{f}", mime="text/html" if f == "html" else "text/markdown")
        if f == "html":
            st.components.v1.html(body, height=800, scrolling=True)
        else:
            st.markdown(body)
