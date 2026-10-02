"""Conversational analytics agent orchestrated with LangGraph.

classify → retrieve (RAG) → {plan → execute ⇄ repair | investigate | anomalies | trend | define} → summarize → verify
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph

from app.core import anomaly, trends, visuals
from app.core.insights import _fmt, pick_date_column, pick_primary_metric
from app.core.profiling import profile_dataframe
from app.data.store import DataStore
from .llm import LLM
from .nl2sql import HeuristicPlanner, MONTHS, classify_intent
from .rag import KnowledgeBase
from .sandbox import SandboxError, run_python
from .schemas import Answer, Intent, QueryPlan
from .verify import verify_numbers

MAX_REPAIRS = 2

SQL_SYSTEM = """You are a senior data analyst. Convert the user's question into ONE read-only SQL SELECT statement.
Rules:
- Use ONLY the tables/columns in the schema. Never invent columns. SQL dialect is given in the schema.
- Prefer explicit JOINs using the listed join keys. Qualify columns with table names.
- Follow the business definitions in the context exactly (e.g. how revenue, margin, active customer are defined).
- Return aggregated results (not raw dumps) unless the user asks for rows. Add ORDER BY and a sensible LIMIT.
- Optionally give `python` (pandas, variable `df` = SQL result, must assign `result`) only if SQL alone cannot answer.
"""
SUMMARY_SYSTEM = """You are a business analyst explaining verified results to an executive.
Use ONLY numbers that appear in the provided evidence — never estimate or invent figures.
Be concise (3-6 sentences). State the direct answer first, then the key driver/pattern, then one recommended action if warranted.
Mention when evidence is limited."""


class LazyTables(Mapping):
    def __init__(self, store: DataStore):
        self.store, self._c = store, {}

    def __getitem__(self, k):
        if k not in self._c:
            self._c[k] = self.store.get_dataframe(k)
        return self._c[k]

    def __iter__(self):
        return iter(self.store.dataset_names())

    def __len__(self):
        return len(self.store.dataset_names())


class State(TypedDict, total=False):
    question: str
    history: List[Dict[str, Any]]
    intent: str
    context: List[Dict[str, Any]]
    plan: Optional[QueryPlan]
    df: Optional[pd.DataFrame]
    error: Optional[str]
    attempts: int
    evidence: Dict[str, Any]
    steps: List[str]
    template: str
    answer: str
    fig: Any
    verification: Dict[str, Any]
    engine: str


def describe_result(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "The query returned no rows — the filters may be too restrictive or the values don't exist in the data."
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dims = [c for c in df.columns if c not in num]
    label = lambda c: str(c).replace("_", " ")
    if len(df) == 1 and num and not dims:
        return "  \n".join(f"**{label(c)}: {_fmt(float(df[c].iloc[0]))}**" for c in num)
    if not num:
        return f"Returned {len(df)} row(s) with columns: {', '.join(map(label, df.columns))}."
    m = num[-1]
    key = " / ".join(dims) if dims else "row"
    names = df[dims].astype(str).agg(" / ".join, axis=1) if dims else df.index.astype(str)
    is_time = dims and (dims[0] in {"day", "week", "month", "quarter", "year"} or pd.api.types.is_datetime64_any_dtype(df[dims[0]]))
    if is_time and len(df) >= 2:
        first, last = float(df[m].iloc[0]), float(df[m].iloc[-1])
        pk, lo = df[m].idxmax(), df[m].idxmin()
        chg = f" ({(last - first) / abs(first) * 100:+.1f}%)" if first else ""
        return (f"**{label(m)}** moved from {_fmt(first)} in {names.iloc[0]} to {_fmt(last)} in {names.iloc[-1]}{chg}. "
                f"Peak: {names[pk]} ({_fmt(float(df[m][pk]))}); low: {names[lo]} ({_fmt(float(df[m][lo]))}).")
    total = float(df[m].sum())
    top = ", ".join(f"{names.iloc[i]} ({_fmt(float(df[m].iloc[i]))})" for i in range(min(3, len(df))))
    txt = f"{label(m).capitalize()} by {key}: {top}."
    if len(df) > 3 and total and df[m].min() >= 0 and "rate" not in m and "avg" not in m:
        txt += f" The top entry is {float(df[m].iloc[0]) / total * 100:.1f}% of the {_fmt(total)} shown across {len(df)} groups."
    if len(df) > 3:
        txt += f" Last in list: {names.iloc[-1]} ({_fmt(float(df[m].iloc[-1]))})."
    return txt


def _records(df: Optional[pd.DataFrame], n: int = 50) -> List[Dict[str, Any]]:
    if df is None:
        return []
    return json.loads(df.head(n).to_json(orient="records", date_format="iso"))


class AnalyticsAgent:
    def __init__(self, store: DataStore, kb: KnowledgeBase, llm: Optional[LLM] = None):
        self.store, self.kb, self.llm = store, kb, llm
        self._planner_key, self._planner = None, None
        self.graph = self._build()

    # ------------------------------------------------------------------ public
    def ask(self, question: str, history: Optional[List[Dict[str, Any]]] = None) -> Answer:
        if not self.store.dataset_names():
            return Answer(question=question, answer="No datasets are loaded yet — upload a CSV/Excel file first.", intent="none",
                          engine="none", error="no_data")
        st: State = self.graph.invoke({"question": question, "history": history or [], "steps": [], "attempts": 0, "evidence": {}})
        df = st.get("df")
        fig = st.get("fig")
        if fig is None and df is not None and not df.empty and st.get("intent") == "query":
            try:
                fig = visuals.auto_chart(df)
            except Exception:
                fig = None
        return Answer(
            question=question, answer=st.get("answer", ""), intent=st.get("intent", "query"),
            engine=self.llm.name if self.llm else "heuristic", sql=st["plan"].sql if st.get("plan") else None,
            columns=list(df.columns) if df is not None else [], rows=_records(df),
            chart_json=fig.to_json() if fig is not None else None, evidence=json.loads(json.dumps(st.get("evidence", {}), default=str)),
            context=[{"title": c.get("title"), "kind": c.get("kind"), "score": c.get("score"), "text": c["text"][:400]} for c in st.get("context", [])],
            verification=st.get("verification", {}), steps=st.get("steps", []), error=st.get("error"))

    # ------------------------------------------------------------------ helpers
    def planner(self) -> HeuristicPlanner:
        ds = self.store.list_datasets()
        key = tuple((d["name"], d["rows"], d["created_at"]) for d in ds)
        if key != self._planner_key:
            schema = self.store.schema()
            values: Dict[Any, Dict[str, str]] = {}
            for t, cols in schema.items():
                for c in cols:
                    if any(k in c["type"].upper() for k in ("CHAR", "TEXT", "STRING", "VARCHAR")):
                        try:
                            v = self.store.run_sql(f"SELECT DISTINCT {c['name']} FROM {t}", limit=60)[c["name"]].dropna().astype(str)
                        except Exception:
                            continue
                        if 0 < len(v) <= 50:
                            values[(t, c["name"])] = {x.lower(): x for x in v}
            self._planner = HeuristicPlanner(schema, self.store.relationships(), {d["name"]: d["rows"] for d in ds},
                                             values, self.store.engine.dialect.name)
            self._planner_key = key
        return self._planner

    def _analysis_frame(self, question: str) -> tuple[pd.DataFrame, Optional[str], Optional[str], str]:
        hit = self.planner().detect_metric(question)
        fact = hit[0] if hit else None
        df = self.store.joined_frame(fact)
        prof = profile_dataframe(df)
        metric = hit[1] if hit and hit[1] in df.columns else pick_primary_metric(prof, df)
        return df, pick_date_column(prof, df), metric, (fact or "")

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(State)
        for name, fn in [("classify", self.n_classify), ("retrieve", self.n_retrieve), ("plan", self.n_plan),
                         ("execute", self.n_execute), ("repair", self.n_repair), ("investigate", self.n_investigate),
                         ("anomalies", self.n_anomalies), ("trend", self.n_trend), ("define", self.n_define),
                         ("summarize", self.n_summarize), ("verify", self.n_verify)]:
            g.add_node(name, fn)
        g.set_entry_point("classify")
        g.add_edge("classify", "retrieve")
        g.add_conditional_edges("retrieve", lambda s: s["intent"] if s["intent"] != "query" else "plan",
                                {"plan": "plan", "investigate": "investigate", "anomalies": "anomalies", "trend": "trend", "define": "define"})
        g.add_edge("plan", "execute")
        g.add_conditional_edges("execute", lambda s: "repair" if s.get("error") and self.llm and s["attempts"] <= MAX_REPAIRS else "summarize",
                                {"repair": "repair", "summarize": "summarize"})
        g.add_edge("repair", "execute")
        for n in ("investigate", "anomalies", "trend", "define"):
            g.add_edge(n, "summarize")
        g.add_edge("summarize", "verify")
        g.add_edge("verify", END)
        return g.compile()

    # ------------------------------------------------------------------ nodes
    def n_classify(self, s: State) -> State:
        intent = classify_intent(s["question"])
        if self.llm:
            try:
                intent = self.llm.structured(Intent, "Classify the analytics question.", s["question"]).intent
            except Exception:
                pass
        return {"intent": intent, "steps": s["steps"] + [f"intent={intent}"]}

    def n_retrieve(self, s: State) -> State:
        ctx = self.kb.search(s["question"], k=4, kind="glossary", min_score=0.12) + self.kb.search(s["question"], k=4, kind="schema", min_score=0.12)
        return {"context": ctx, "steps": s["steps"] + [f"rag: retrieved {len(ctx)} context chunks"]}

    def n_plan(self, s: State) -> State:
        steps = list(s["steps"])
        plan = None
        if self.llm:
            ctx = "\n".join(f"- {c['text']}" for c in s["context"])
            hist = "\n".join(f"Q: {h['question']}\nSQL: {h.get('sql')}" for h in s["history"][-3:])
            try:
                plan = self.llm.structured(QueryPlan, SQL_SYSTEM, f"SCHEMA:\n{self.store.schema_text()}\n\nBUSINESS CONTEXT:\n{ctx}\n\n"
                                                        f"RECENT CONVERSATION:\n{hist}\n\nQUESTION: {s['question']}")
                steps.append("plan: LLM generated SQL")
            except Exception as e:
                steps.append(f"plan: LLM failed ({type(e).__name__}); using heuristic planner")
        if plan is None:
            plan = self.planner().plan(s["question"])
            steps.append("plan: heuristic planner" + ("" if plan else " could not interpret the question"))
        if plan is None:
            return {"plan": None, "error": "I couldn't translate that into a query. Try naming a metric and a grouping, e.g. "
                    "'total revenue by region' — or configure an LLM provider for free-form questions.", "steps": steps, "attempts": MAX_REPAIRS + 1}
        return {"plan": plan, "steps": steps}

    def n_execute(self, s: State) -> State:
        plan = s.get("plan")
        if plan is None:
            return {}
        steps = list(s["steps"])
        try:
            df = self.store.run_sql(plan.sql)
            steps.append(f"execute: {len(df)} rows")
            if plan.python:
                out = run_python(plan.python, df=df, tables=LazyTables(self.store))
                df = out if isinstance(out, pd.DataFrame) else pd.DataFrame(out) if isinstance(out, (dict, list)) else \
                    out.reset_index() if isinstance(out, pd.Series) else pd.DataFrame({"result": [out]})
                steps.append("python: post-processed in sandbox")
            return {"df": df, "error": None, "steps": steps, "evidence": {"sql": plan.sql, "explanation": plan.explanation}}
        except Exception as e:
            steps.append(f"execute: error {type(e).__name__}: {str(e)[:120]}")
            return {"error": str(e)[:500], "attempts": s["attempts"] + 1, "steps": steps}

    def n_repair(self, s: State) -> State:
        try:
            plan = self.llm.structured(QueryPlan, SQL_SYSTEM, f"SCHEMA:\n{self.store.schema_text()}\n\nQUESTION: {s['question']}\n\n"
                                       f"PREVIOUS SQL:\n{s['plan'].sql}\n\nERROR:\n{s['error']}\n\nReturn a corrected query.")
            return {"plan": plan, "steps": s["steps"] + ["repair: LLM revised SQL"]}
        except Exception:
            return {"attempts": MAX_REPAIRS + 1}

    def n_investigate(self, s: State) -> State:
        df, date_col, metric, fact = self._analysis_frame(s["question"])
        if not date_col or not metric:
            return {"template": "Root-cause analysis needs a date column and a numeric metric; I couldn't find both.", "steps": s["steps"]}
        prof = profile_dataframe(df)
        dims = [c["name"] for c in prof["column_profiles"] if c["semantic_type"] == "categorical"][:8]
        cur = prev = None
        q = s["question"].lower()
        mon = next((i + 1 for i, m in enumerate(MONTHS) if re.search(rf"\b{m}\b", q)), None)
        if mon:
            ym = re.search(r"\b(20\d\d)\b", q)
            year = int(ym.group(1)) if ym else int(df[date_col].max().year)
            start = pd.Timestamp(year=year, month=mon, day=1)
            pstart = start - pd.offsets.MonthBegin(1)
            cur, prev = (start, start + pd.offsets.MonthEnd(0)), (pstart, pstart + pd.offsets.MonthEnd(0))
        drv = trends.driver_analysis(df, date_col, metric, dims, cur, prev)
        if "error" in drv or not drv.get("previous_total"):
            return {"template": f"I couldn't compare periods for {metric}: {drv.get('error', 'no data in the previous period')}.", "steps": s["steps"]}
        hs, top = drv.get("hotspot"), (drv["drivers"][0] if drv["drivers"] else None)
        direction = "rose" if drv["delta"] > 0 else "fell"
        txt = (f"**{metric.replace('_', ' ').title()} {direction} {abs(drv['pct_change']):.1f}%** ({_fmt(drv['delta'])}), from {_fmt(drv['previous_total'])} "
               f"({drv['previous_period'][0]} → {drv['previous_period'][1]}) to {_fmt(drv['current_total'])} "
               f"({drv['current_period'][0]} → {drv['current_period'][1]}).")
        if top:
            txt += f" Biggest single driver: **{top['dimension']} = {top['member']}** ({_fmt(top['delta'])}, {top['share_of_change_pct']}% of the change)."
        if hs:
            cell = " & ".join(f"{k} = {v}" for k, v in hs["cell"].items())
            txt += (f" The change is concentrated in **{cell}**: it explains {hs['share_of_change_pct']}% of the movement while being only "
                    f"{hs['base_share_pct']}% of the previous period ({_fmt(hs['previous'])} → {_fmt(hs['current'])}). "
                    f"Recommended next step: check operations, pricing, stock or tracking changes for that segment.")
        fig = None
        if drv["drivers"]:
            fig = visuals.waterfall(drv["drivers"][:5], drv["previous_total"], drv["current_total"])
        return {"template": txt, "evidence": drv, "fig": fig, "df": pd.DataFrame(drv["drivers"]) if drv["drivers"] else None,
                "steps": s["steps"] + [f"investigate: driver analysis on {metric} ({fact or 'joined data'})"]}

    def n_anomalies(self, s: State) -> State:
        df, date_col, metric, fact = self._analysis_frame(s["question"])
        parts, ev, fig = [], {}, None
        if date_col and metric:
            daily = trends.aggregate_series(df, date_col, metric, "sum", "D")
            ta = anomaly.time_series_anomalies(daily)
            if len(ta):
                ev["daily"] = ta.head(8).reset_index().astype({ta.index.name or "index": str}).to_dict(orient="records")
                t = ta.iloc[0]
                parts.append(f"Found {len(ta)} anomalous day(s) in daily {metric}. Most extreme: **{ta.index[0].date()}** — a {t['direction']} of "
                             f"{_fmt(t['value'])} versus an expected {_fmt(t['expected'])} ({t['deviation_pct']:+.0f}%).")
                fig = visuals.time_series(daily, f"Daily {metric} with anomalies", ta)
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_id")]
        ua = anomaly.univariate_anomalies(df, num)
        for f in ua["findings"][:3]:
            ev.setdefault("outliers", []).append({k: v for k, v in f.items() if k not in ("examples", "index")})
            parts.append(f"'{f['column']}' has {f['count']} outlier row(s) ({f['pct_of_rows']}%).")
        if not parts:
            parts.append("No statistically significant anomalies were detected in the main metric or numeric columns.")
        return {"template": " ".join(parts), "evidence": ev, "fig": fig, "steps": s["steps"] + ["anomalies: robust-z + rolling-median residuals"]}

    def n_trend(self, s: State) -> State:
        df, date_col, metric, _ = self._analysis_frame(s["question"])
        if not date_col or not metric:
            return {"template": "Trend analysis needs a date column and a numeric metric; I couldn't find both."}
        ser = trends.aggregate_series(df, date_col, metric, "sum")
        tr = trends.detect_trend(ser)
        if tr["direction"] == "insufficient_data":
            return {"template": "There isn't enough history to establish a trend."}
        txt = (f"**{metric.replace('_', ' ').title()}** is trending **{tr['direction']}**: {_fmt(tr['first_value'])} → {_fmt(tr['last_value'])} "
               f"({tr['pct_change_first_last']:+.1f}%) from {tr['start']} to {tr['end']} (R²={tr['r_squared']}, p={tr['p_value']:.3g}).")
        return {"template": txt, "evidence": tr, "fig": visuals.time_series(ser, f"{metric} over time"),
                "df": ser.reset_index().rename(columns={date_col: "period"}), "steps": s["steps"] + ["trend: linear regression on aggregated series"]}

    def n_define(self, s: State) -> State:
        hits = [c for c in s["context"] if c.get("kind") == "glossary"][:2]
        if len(hits) == 2 and hits[1]["score"] < 0.8 * hits[0]["score"]:
            hits = hits[:1]
        if not hits:
            return {"template": "I couldn't find a matching definition in the business glossary. Upload a glossary (Markdown) to enable this.", "steps": s["steps"]}
        return {"template": "\n\n".join(h["text"] for h in hits), "evidence": {"definitions": [h["title"] for h in hits]}, "steps": s["steps"]}

    def n_summarize(self, s: State) -> State:
        if s.get("error") and s.get("df") is None:
            return {"answer": f"⚠️ {s['error']}"}
        template = s.get("template") or describe_result(s.get("df"))
        answer, steps = template, list(s["steps"])
        if self.llm and s["intent"] != "define":
            try:
                ev = {"question": s["question"], "sql": s.get("plan").sql if s.get("plan") else None,
                      "result_rows": _records(s.get("df"), 30), "analysis": s.get("evidence", {}), "context": [c["text"] for c in s.get("context", [])][:4]}
                answer = self.llm.text(SUMMARY_SYSTEM, json.dumps(ev, default=str)[:24000])
                steps.append("summarize: LLM narrative")
            except Exception:
                answer = template
        return {"answer": answer, "steps": steps}

    def n_verify(self, s: State) -> State:
        pool = [s.get("evidence", {})]
        df = s.get("df")
        if df is not None and not df.empty:
            pool.append(_records(df, 200))
            for c in df.select_dtypes("number").columns:
                col = df[c].astype(float)
                tot = float(col.sum())
                pool.append({"sum": tot, "mean": float(col.mean()),
                             "shares": [float(v) / tot * 100 for v in col] if tot else [],
                             "chg_first_last": (float(col.iloc[-1]) - float(col.iloc[0])) / abs(float(col.iloc[0])) * 100 if len(col) > 1 and col.iloc[0] else 0,
                             "step_changes": [(float(b) - float(a)) / abs(float(a)) * 100 for a, b in zip(col[:-1], col[1:]) if a]})
        v = verify_numbers(s.get("answer", ""), pool)
        ans = s.get("answer", "")
        if not v["verified"] and self.llm:
            ans += f"\n\n⚠️ Some figures could not be matched to computed results: {', '.join(v['unverified'][:5])}"
        return {"verification": v, "answer": ans}
