"""Restricted execution of LLM-written pandas code.

Defence in depth (AST allow-checks, stripped builtins, line-event budget). It is meant to stop accidents and
prompt-injection attempts; for hostile multi-tenant deployments additionally run workers in a locked-down container.
"""
from __future__ import annotations

import ast
import sys
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars", "dir",
                 "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit", "memoryview", "type"}
ALLOWED_TO = {"to_dict", "to_list", "to_numpy", "to_datetime", "to_period", "to_timestamp", "to_frame",
              "to_string", "to_series", "to_numeric", "to_timedelta", "tolist"}
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ["abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "map", "max",
                           "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip", "filter", "reversed",
                           "isinstance", "print", "pow", "divmod", "slice", "True", "False", "None"]}


class SandboxError(Exception):
    pass


def validate_code(code: str) -> ast.AST:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxError(f"Syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise SandboxError("Imports and async/global statements are not allowed (pd, np, stats are preloaded)")
        if isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            raise SandboxError(f"Use of '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_"):
                raise SandboxError(f"Access to private attribute '{a}' is not allowed")
            if a.startswith("read_") or (a.startswith("to_") and a not in ALLOWED_TO) or a in {"eval", "query_file", "system"}:
                raise SandboxError(f"Method '{a}' is not allowed (file/IO access)")
    return tree


class _Budget:
    def __init__(self, max_events: int):
        self.left = max_events

    def __call__(self, frame, event, arg):
        if event == "line":
            self.left -= 1
            if self.left <= 0:
                raise SandboxError("Execution budget exceeded (possible infinite loop)")
        return self


def run_python(code: str, df: Optional[pd.DataFrame] = None, tables: Optional[Mapping[str, pd.DataFrame]] = None,
               max_line_events: int = 200_000) -> Any:
    """Execute `code`; it must assign a variable called `result`. Returns that value."""
    tree = validate_code(code)
    env: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, "stats": stats,
                           "df": df.copy() if df is not None else None, "tables": tables or {}}
    budget = _Budget(max_line_events)
    old = sys.gettrace()
    sys.settrace(budget)
    try:
        exec(compile(tree, "<analysis>", "exec"), env, env)  # noqa: S102 - validated above
    except SandboxError:
        raise
    except Exception as e:
        raise SandboxError(f"{type(e).__name__}: {e}") from e
    finally:
        sys.settrace(old)
    if "result" not in env:
        raise SandboxError("Code must assign its output to a variable named `result`")
    return env["result"]
