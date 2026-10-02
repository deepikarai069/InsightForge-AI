"""Read-only SQL validation. LLM-generated SQL is *never* executed without passing through here."""
from __future__ import annotations

import re
from typing import Iterable, Set

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|attach|detach|pragma|copy|merge|replace|"
    r"vacuum|call|exec|execute|into|set|reindex|analyze|load_extension|pg_sleep|lo_import|lo_export)\b", re.I)
_COMMENTS = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.S)
_TABLE_REF = re.compile(r"\b(?:from|join)\s+(\"?[A-Za-z_][\w\.\"]*)", re.I)
_CTE = re.compile(r"\b([A-Za-z_]\w*)\s+as\s*\(", re.I)


class SQLValidationError(ValueError):
    pass


def _strip_string_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_sql(sql: str, allowed_tables: Iterable[str]) -> str:
    """Return a cleaned single SELECT statement or raise SQLValidationError."""
    if not sql or not sql.strip():
        raise SQLValidationError("Empty SQL")
    cleaned = _COMMENTS.sub(" ", sql).strip().rstrip(";").strip()
    if ";" in _strip_string_literals(cleaned):
        raise SQLValidationError("Multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.I):
        raise SQLValidationError("Only SELECT queries are allowed")
    body = _strip_string_literals(cleaned)
    m = FORBIDDEN.search(body)
    if m:
        raise SQLValidationError(f"Forbidden keyword: {m.group(1).upper()}")
    allowed: Set[str] = {t.lower() for t in allowed_tables}
    ctes = {c.lower() for c in _CTE.findall(body)}
    for ref in _TABLE_REF.findall(body):
        name = ref.strip('"').lower()
        if name not in allowed and name not in ctes:
            raise SQLValidationError(f"Unknown or disallowed table: {ref}")
    return cleaned
