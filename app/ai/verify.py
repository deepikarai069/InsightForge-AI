"""Evidence verification: every number in a narrative answer must trace back to computed results."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Set

_NUM = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s*([KMB%]?)(?![\w])")
_MULT = {"K": 1e3, "M": 1e6, "B": 1e9}


def _flatten(obj: Any, out: List[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if not (isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj))):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)
    elif isinstance(obj, str):
        for m in _NUM.finditer(obj):
            try:
                out.append(float(m.group(1).replace(",", "")))
            except ValueError:
                pass


def _close(a: float, b: float, rel: float = 0.02, abs_tol: float = 0.06) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def verify_numbers(text: str, evidence: Any, ignore_below: float = 10) -> Dict[str, Any]:
    """Return which numbers in `text` could (not) be matched with numbers in `evidence`.

    Matches directly, as a percentage of 100, K/M/B scaled, or absolute value (for reported changes).
    """
    pool: List[float] = []
    _flatten(evidence, pool)
    checked, unverified = 0, []
    for m in _NUM.finditer(text):
        raw, suf = m.group(1), m.group(2)
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if 1900 <= abs(v) <= 2100 and not suf and "," not in raw and "." not in raw:
            continue  # looks like a year
        if abs(v) < ignore_below and not suf and "." not in raw:
            continue  # small counting integers ("top 5", "3 regions")
        value = v * _MULT.get(suf, 1)
        cands = {value, abs(value), v, abs(v)}
        if suf == "%":
            cands |= {v / 100, abs(v) / 100}
        checked += 1
        if not any(_close(c, p) or _close(c, abs(p)) or _close(c, p * 100) for c in cands for p in pool):
            unverified.append(m.group(0).strip())
    return {"numbers_checked": checked, "unverified": unverified, "verified": not unverified}
