"""Deterministic natural-language → SQL planner (offline fallback) and intent classifier.

It covers the common analytic questions (aggregations, group-bys, top-N, time grains, value filters, one/two-hop joins,
AOV / rate / margin metrics). With an LLM configured the agent uses the LLM planner instead and only falls back here.
"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from .schemas import QueryPlan

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
SYN = {"sales": "revenue", "income": "revenue", "turnover": "revenue", "units": "quantity", "spend": "amount", "cost": "unit_cost"}
NUMERIC_T = ("INT", "FLOAT", "REAL", "NUMERIC", "DOUBLE", "DECIMAL", "BIGINT", "NUMBER")
TIME_WORDS = {"day": "day", "daily": "day", "week": "week", "weekly": "week", "month": "month", "monthly": "month",
              "quarter": "quarter", "quarterly": "quarter", "year": "year", "yearly": "year", "annual": "year", "annually": "year"}


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def classify_intent(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(why|reason|root cause|cause|driver|drivers|what happened|explain (the )?(drop|decline|increase|spike|change)|"
                 r"(dropped|declined|decreased|fell|fall|down|decline|drop|surged|jumped|increased))\b", q) and \
            not re.search(r"\b(top|list)\b", q):
        return "investigate"
    if re.search(r"\b(anomal\w*|outlier\w*|unusual|abnormal|suspicious|strange|weird|spike days?)\b", q):
        return "anomalies"
    if re.search(r"\b(trend|trending|trajectory|growth|growing|over time|seasonal\w*)\b", q):
        return "trend"
    analytic = re.search(r"\b(by|per|each|total|sum|average|avg|count|top|bottom|how many|in (20\d\d|january|february|march|april|may|june|july|august|september|october|november|december))\b", q)
    if not analytic and (re.search(r"^(what is|what's|what does|define|definition of|meaning of|how (do|is|are) we (calculate|compute|define|measure)|"
                                   r"how is .* (calculated|defined|computed))", q.strip()) or re.search(r"\b(definition|glossary|means?)\b", q)):
        return "define"
    return "query"


class HeuristicPlanner:
    def __init__(self, schema: Dict[str, List[Dict[str, Any]]], relationships: List[Dict[str, Any]],
                 row_counts: Dict[str, int], values: Dict[Tuple[str, str], Dict[str, str]], dialect: str = "sqlite"):
        self.schema = schema
        self.rels = relationships
        self.rows = row_counts
        self.values = values           # (table, col) -> {lower_value: original_value}
        self.dialect = dialect
        self.cols: List[Tuple[str, str, str]] = [(t, c["name"], c["type"].upper()) for t, cs in schema.items() for c in cs]

    # ---- helpers ---------------------------------------------------------
    def is_num(self, t: str) -> bool:
        return any(k in t for k in NUMERIC_T)

    def is_date(self, t: str) -> bool:
        return "DATE" in t or "TIME" in t

    def date_col(self, table: str) -> Optional[str]:
        c = [n for tb, n, ty in self.cols if tb == table and self.is_date(ty)]
        c.sort(key=lambda n: (0 if "order" in n or "created" in n else 1, n))
        return c[0] if c else None

    def id_col(self, table: str) -> Optional[str]:
        stem = _stem(table)
        cols = [n for tb, n, _ in self.cols if tb == table and n.endswith("_id")]
        for n in cols:
            if n == f"{stem}_id":
                return n
        return cols[0] if cols else None

    def _tokens(self, q: str) -> List[str]:
        toks = re.findall(r"[a-z0-9_]+", q.lower())
        return [SYN.get(t, t) for t in toks]

    def detect_metric(self, question: str) -> Optional[Tuple[str, str]]:
        toks = {_stem(t) for t in self._tokens(question)}
        best, best_score = None, 0.0
        for t, n, ty in self.cols:
            if not self.is_num(ty) or n.endswith("_id"):
                continue
            parts = {_stem(p) for p in n.split("_")}
            score = len(parts & toks) / len(parts) if parts & toks else 0.0
            if _stem(n) in toks:
                score += 1
            score += 0.05 * (self.rows.get(t, 0) > 0) + 0.02 * (self.rows.get(t, 0) / (1 + max(self.rows.values(), default=1)))
            if score > best_score and (parts & toks or _stem(n) in toks):
                best, best_score = (t, n), score
        return best

    def _find_dim(self, word: str, prefer: Optional[str]) -> Optional[Tuple[str, str]]:
        w, ws = word.strip().lower(), _stem(word.strip().lower())
        cands = []
        for t, n, ty in self.cols:
            nl = n.lower()
            if nl in (w, ws):
                rank = 0
            elif nl.startswith(ws + "_") or nl.startswith(w + "_"):
                rank = 1 if nl.endswith("_name") else 2
            elif ws in nl.split("_"):
                rank = 3
            else:
                continue
            cands.append((rank, 0 if t == prefer else 1, -self.rows.get(t, 0), t, n))
        if not cands:
            return None
        cands.sort()
        return cands[0][3], cands[0][4]

    def _path(self, src: str, dst: str) -> Optional[List[Tuple[str, str, str, str]]]:
        """BFS join path from src table to dst table as list of (t1, c1, t2, c2)."""
        if src == dst:
            return []
        adj: Dict[str, List[Tuple[str, str, str, str]]] = {}
        for r in self.rels:
            adj.setdefault(r["left_table"], []).append((r["left_table"], r["left_column"], r["right_table"], r["right_column"]))
            adj.setdefault(r["right_table"], []).append((r["right_table"], r["right_column"], r["left_table"], r["left_column"]))
        seen, dq = {src}, deque([(src, [])])
        while dq:
            cur, path = dq.popleft()
            for e in adj.get(cur, []):
                if e[2] in seen:
                    continue
                if e[2] == dst:
                    return path + [e]
                seen.add(e[2])
                dq.append((e[2], path + [e]))
        return None

    def _year(self, col: str) -> str:
        return f"strftime('%Y', {col})" if self.dialect == "sqlite" else f"to_char({col}, 'YYYY')"

    def _time_expr(self, col: str, grain: str) -> str:
        sq = self.dialect == "sqlite"
        return {
            "day": f"date({col})" if sq else f"to_char({col}, 'YYYY-MM-DD')",
            "week": f"strftime('%Y-W%W', {col})" if sq else f"to_char({col}, 'IYYY-\"W\"IW')",
            "month": f"strftime('%Y-%m', {col})" if sq else f"to_char({col}, 'YYYY-MM')",
            "quarter": (f"strftime('%Y', {col}) || '-Q' || ((cast(strftime('%m', {col}) as integer) + 2) / 3)" if sq
                        else f"to_char({col}, 'YYYY-\"Q\"Q')"),
            "year": self._year(col),
        }[grain]

    # ---- main --------------------------------------------------------------
    def plan(self, question: str) -> Optional[QueryPlan]:
        q = question.lower().strip().rstrip("?.! ")
        toks = self._tokens(q)
        tokset = set(toks)
        joins_needed: Set[str] = set()
        expr, alias, explain_metric = None, None, None
        fact: Optional[str] = None

        # --- metric ---------------------------------------------------------
        metric = self.detect_metric(q)
        count_intent = bool(re.search(r"\b(how many|number of|count|no\. of)\b", q))
        rev = next(((t, n) for t, n, ty in self.cols if n == "revenue"), None) or self.detect_metric("revenue sales amount")
        if re.search(r"\b(aov|average order value|avg order value|average basket)\b", q) and rev:
            oid = next(((t, n) for t, n, _ in self.cols if t == rev[0] and n.endswith("order_id")), None)
            if oid:
                fact = rev[0]
                expr = f"SUM({fact}.{rev[1]}) * 1.0 / COUNT(DISTINCT {fact}.{oid[1]})"
                alias, explain_metric = "average_order_value", "average order value (revenue ÷ distinct orders)"
        if expr is None and re.search(r"\bgross margin\b", q) and rev:
            cost = next(((t, n) for t, n, _ in self.cols if n == "unit_cost"), None)
            qty = next(((t, n) for t, n, _ in self.cols if t == rev[0] and n == "quantity"), None)
            if cost and qty:
                fact = rev[0]
                joins_needed.add(cost[0])
                pct = bool(re.search(r"%|percent|margin rate|margin pct", q))
                cogs = f"SUM({fact}.{qty[1]} * {cost[0]}.{cost[1]})"
                expr = f"(SUM({fact}.{rev[1]}) - {cogs}) * 100.0 / SUM({fact}.{rev[1]})" if pct else f"SUM({fact}.{rev[1]}) - {cogs}"
                alias, explain_metric = ("gross_margin_pct" if pct else "gross_margin"), "gross margin (revenue − quantity × unit_cost)"
        if expr is None and re.search(r"\b(rate|ratio|share|percentage|percent|proportion)\b", q):
            for t in toks:
                st = _stem(t)
                if len(st) < 4 or st in {"rate", "ratio", "share", "percentage", "percent", "proportion", "order", "customer", "region"}:
                    continue
                hit = next(((tb, n) for tb, n, ty in self.cols if self.is_num(ty) and not n.endswith("_id") and n.startswith(st[:5])), None)
                if hit:
                    fact = hit[0]
                    expr = f"AVG({hit[0]}.{hit[1]} * 1.0)"
                    alias, explain_metric = f"{hit[1]}_rate", f"average of {hit[1]} (rate)"
                    break
        if expr is None and count_intent:
            tgt = None
            for t in toks:
                for tb in self.schema:
                    if _stem(t) == _stem(tb) or _stem(t) == _stem(tb.rstrip("s")):
                        tgt = tb
            if tgt is None and metric is None:
                tgt = max(self.rows, key=self.rows.get) if self.rows else None
            if tgt:
                fact = tgt
                idc = self.id_col(tgt)
                distinct = bool(re.search(r"distinct|unique", q)) or (idc and _stem(tgt) in (idc[:-3], ) and idc is not None)
                expr = f"COUNT(DISTINCT {tgt}.{idc})" if idc and distinct else "COUNT(*)"
                alias, explain_metric = f"{_stem(tgt)}_count", f"number of {tgt}"
        if expr is None and metric:
            fact, col = metric
            if re.search(r"\b(average|avg|mean|typical)\b", q) or re.search(r"price|rate|discount|score|rating|margin", col):
                agg, alias = "AVG", f"avg_{col}"
            elif re.search(r"\b(max|maximum|highest|largest|biggest)\b", q) and not re.search(r"\b(by|per|each)\b", q):
                agg, alias = "MAX", f"max_{col}"
            elif re.search(r"\b(min|minimum|lowest|smallest)\b", q) and not re.search(r"\b(by|per|each)\b", q):
                agg, alias = "MIN", f"min_{col}"
            else:
                agg, alias = "SUM", f"total_{col}"
            expr = f"{agg}({fact}.{col})"
            explain_metric = f"{agg.lower()} of {col}"
        if expr is None or fact is None:
            return None

        # --- dimensions ---------------------------------------------------------
        dims: List[Tuple[str, str, str]] = []   # (select_expr, alias, table)
        time_grain = None
        dim_words: List[str] = []
        for m in re.finditer(r"\b(?:by|per|each|across|for every|grouped by)\s+([a-z_ ,]+?)(?=\s+(?:in|for|during|over|last|where|with|from|since|between|of)\b|$|[?.])", q):
            dim_words += [w.strip() for w in re.split(r",|\band\b", m.group(1)) if w.strip()]
        top_n = None
        mt = re.search(r"\b(?:top|bottom|lowest|highest|best|worst)\s+(\d+)\s+([a-z_]+)", q)
        mt2 = None if mt else re.search(r"\b(?:top|bottom|lowest|highest|best|worst)\s+([a-z_]+)\s+by\b", q)
        if mt:
            top_n = int(mt.group(1))
            dim_words = [mt.group(2)] + [w for w in dim_words if self.detect_metric(w) is None]
        elif mt2 and mt2.group(1) not in ("n",):
            dim_words = [mt2.group(1)] + [w for w in dim_words if self.detect_metric(w) is None]
        for mm in re.finditer(r"\b(monthly|weekly|daily|quarterly|yearly|annually)\b", q):
            dim_words.append(mm.group(1))
        seen_alias: Set[str] = set()
        for w in dim_words:
            w1 = w.split()[-1] if w.split() else w
            if w1 in TIME_WORDS:
                dc = self.date_col(fact) or next((n for t, n, ty in self.cols if self.is_date(ty)), None)
                if dc:
                    dt_table = fact if self.date_col(fact) else next(t for t, n, ty in self.cols if n == dc)
                    joins_needed.add(dt_table)
                    time_grain = TIME_WORDS[w1]
                    if time_grain not in seen_alias:
                        dims.append((self._time_expr(f"{dt_table}.{dc}", time_grain), time_grain, dt_table))
                        seen_alias.add(time_grain)
                continue
            hit = self._find_dim(w1, fact)
            if hit and hit[1] not in seen_alias and hit[1] != (metric[1] if metric else None):
                dims.append((f"{hit[0]}.{hit[1]}", hit[1], hit[0]))
                joins_needed.add(hit[0])
                seen_alias.add(hit[1])

        # --- filters ----------------------------------------------------------------
        where: List[str] = []
        group_cols = {d[1] for d in dims}
        for (t, c), vmap in self.values.items():
            if c in group_cols:
                continue
            for low, orig in vmap.items():
                if len(low) >= 3 and not low.isdigit() and re.search(rf"\b{re.escape(low)}\b", q):
                    where.append(f"{t}.{c} = '{orig.replace(chr(39), chr(39)*2)}'")
                    joins_needed.add(t)
                    break
        ym = re.search(r"\b(20\d\d)\b", q)
        mon = next((i + 1 for i, mn in enumerate(MONTHS) if re.search(rf"\b(?:in|during|for|of)\s+{mn}\b", q)), None)
        dcol = self.date_col(fact)
        if dcol and (ym or mon):
            dc = f"{fact}.{dcol}"
            if ym:
                where.append(f"{self._year(dc)} = '{ym.group(1)}'")
            if mon:
                where.append((f"CAST(strftime('%m', {dc}) AS INTEGER) = {mon}" if self.dialect == "sqlite" else f"EXTRACT(MONTH FROM {dc}) = {mon}"))

        # --- assemble joins ---------------------------------------------------------
        join_sql: List[str] = []
        joined = {fact}
        for t in sorted(joins_needed - {fact}):
            path = self._path(fact, t)
            if path is None:
                return None
            for (t1, c1, t2, c2) in path:
                if t2 not in joined:
                    join_sql.append(f"JOIN {t2} ON {t1}.{c1} = {t2}.{c2}")
                    joined.add(t2)
        sel = [f"{e} AS {a}" for e, a, _ in dims] + [f"{expr} AS {alias}"]
        sql = f"SELECT {', '.join(sel)}\nFROM {fact}"
        if join_sql:
            sql += "\n" + "\n".join(join_sql)
        if where:
            sql += "\nWHERE " + " AND ".join(where)
        if dims:
            sql += "\nGROUP BY " + ", ".join(str(i + 1) for i in range(len(dims)))
            metric_pos = len(dims) + 1
            asc = bool(re.search(r"\b(bottom|lowest|worst|least|smallest|fewest)\b", q))
            if time_grain:
                sql += "\nORDER BY 1"
            else:
                sql += f"\nORDER BY {metric_pos} {'ASC' if asc else 'DESC'}"
                sql += f"\nLIMIT {top_n or 20}"
        expl = f"Computes {explain_metric}" + (f" by {', '.join(a for _, a, _ in dims)}" if dims else "") + \
               (f" where {' and '.join(where)}" if where else "") + "."
        return QueryPlan(sql=sql, explanation=expl)
