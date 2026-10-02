# Architecture

```
 Streamlit UI ─┐                       ┌──────── core/ (Pandas · NumPy · SciPy · scikit-learn)
               ├─► InsightForgeService ┤         cleaning · profiling · stats · trends · anomaly · kpi · insights · visuals
 FastAPI ──────┘        │              └──────── reports/ (HTML + Markdown, Plotly)
                        │
                        ├─► data/  SQLAlchemy store (SQLite | PostgreSQL) + read-only SQL guard
                        └─► ai/    LangGraph agent · NL→SQL · RAG (ChromaDB) · sandbox · number verification
```

## Agent graph (LangGraph)
`classify → retrieve(RAG) → ┬ plan → execute ⇄ repair ┐`
`                           ├ investigate (driver + hotspot analysis) ├→ summarize → verify → END`
`                           ├ anomalies · trend · define ┘`

* **classify** – LLM structured output (`Intent`) or rule-based fallback.
* **retrieve** – semantic search over glossary + schema docs in ChromaDB.
* **plan** – LLM (`QueryPlan` structured output, schema + RAG context + chat history) or deterministic planner.
* **execute / repair** – SQL passes `sql_guard` (single SELECT, table allow-list, forbidden keywords, row cap, read-only transaction on PostgreSQL). On error the LLM gets the error and retries up to 2×.
* **investigate** – decomposes a period-over-period change by dimension, then finds the 2-D "hotspot" cell with the highest concentration lift.
* **verify** – every number in the final narrative is matched against computed evidence; unmatched figures are flagged.

## Safety model
| Risk | Control |
|---|---|
| Destructive / exfiltrating SQL | `sql_guard` allow-list + read-only transaction + statement timeout |
| LLM-written Python | AST validation, no imports/IO/dunder access, stripped builtins, line budget |
| Hallucinated numbers | Evidence verification step; the summariser only receives computed results |
| Silent data changes | Every cleaning action logged and shown in UI/report |

Known limits: the sandbox is defence-in-depth, not a security boundary for hostile multi-tenant use (run workers in a container);
the default hashing embedder is lexical, not semantic — set `EMBEDDING_PROVIDER=openai` for true semantic search.
