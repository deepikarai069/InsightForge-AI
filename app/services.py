"""InsightForgeService — the single façade used by both the FastAPI backend and the Streamlit UI."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.ai.agent import AnalyticsAgent
from app.ai.llm import LLM, get_llm
from app.ai.rag import KnowledgeBase
from app.ai.schemas import Answer
from app.config import Settings, get_settings
from app.core.cleaning import clean_dataframe
from app.core.insights import run_auto_analysis
from app.core.profiling import profile_dataframe
from app.data.loader import load_file
from app.data.store import DataStore
from app.reports.generator import build_report


class InsightForgeService:
    def __init__(self, settings: Optional[Settings] = None, store: Optional[DataStore] = None,
                 kb: Optional[KnowledgeBase] = None, llm: Optional[LLM] = None, use_env_llm: bool = True):
        self.settings = settings or get_settings()
        self.store = store or DataStore(self.settings)
        self.kb = kb or KnowledgeBase(self.settings)
        self.llm = llm if llm is not None else (get_llm(self.settings) if use_env_llm else None)
        self.agent = AnalyticsAgent(self.store, self.kb, self.llm)
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._reports: Dict[str, str] = {}
        self._sync_glossary()

    def _sync_glossary(self) -> None:
        for g in self.store.list_glossary():
            self.kb.add_document(g["title"], g["body"], "glossary")
        self.kb.replace_schema_docs(self.store.schema())

    # ---- ingestion ------------------------------------------------------------
    def ingest_file(self, filename: str, content: bytes, clean: bool = True) -> List[Dict[str, Any]]:
        out = []
        for name, df in load_file(filename, content).items():
            raw_rows = len(df)
            log: List[Dict] = []
            if clean:
                df, cl = clean_dataframe(df)
                log = cl.to_list()
            self.store.register_dataset(name, df, log, filename)
            prof = profile_dataframe(df, name)
            out.append({"name": name, "raw_rows": raw_rows, "rows": len(df), "columns": df.shape[1],
                        "quality_score": prof["quality_score"], "cleaning_actions": log})
        self.kb.replace_schema_docs(self.store.schema())
        return out

    def add_glossary(self, title: str, text: str) -> int:
        self.store.add_glossary(title, text)
        return self.kb.add_document(title, text, "glossary")

    # ---- analysis ---------------------------------------------------------------
    def profile(self, name: str) -> Dict[str, Any]:
        return profile_dataframe(self.store.get_dataframe(name), name)

    def auto_analysis(self, fact: Optional[str] = None, tables: Optional[List[str]] = None) -> Dict[str, Any]:
        df = self.store.joined_frame(fact, tables)
        fact_name = fact or max((d for d in self.store.list_datasets() if not tables or d["name"] in tables), key=lambda d: d["rows"])["name"]
        res = run_auto_analysis(df, fact_name, self.store.get_cleaning_log(fact_name))
        res["frame_columns"] = list(df.columns)
        res["relationships"] = self.store.relationships()
        return res

    def ask(self, question: str, session_id: str = "default") -> Answer:
        hist = self.sessions.setdefault(session_id, [])
        ans = self.agent.ask(question, hist)
        hist.append({"question": question, "sql": ans.sql, "answer": ans.answer})
        del hist[:-10]
        return ans

    def run_sql(self, sql: str, limit: int = 200) -> pd.DataFrame:
        return self.store.run_sql(sql, limit)

    def report(self, fact: Optional[str] = None, title: str = "InsightForge AI — Analytics Report", fmt: str = "html") -> str:
        res = self.auto_analysis(fact)
        df = self.store.joined_frame(fact)
        return build_report(res, df, title, fmt, self.llm)
