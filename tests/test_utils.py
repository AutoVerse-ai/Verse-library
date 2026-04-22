# NOTE: file for utility functions, currently only houses functions for test_reach
import sys
import plotly.graph_objects as go
import os
import json
from typing import Any, Optional

def normalize(obj: Any, float_precision: Optional[int] = None):
    if isinstance(obj, dict):
        return {k: normalize(v, float_precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v, float_precision) for v in obj]
    if isinstance(obj, float) and float_precision is not None:
        return round(obj, float_precision)
    return obj

def canonical_json_bytes(obj: Any, float_precision: Optional[int] = 1e-7) -> bytes:
    norm = normalize(obj, float_precision)
    s = json.dumps(norm, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return s.encode('utf-8')

def json_diffs(a, b, tol=1e-7, path=""):
    # NOTE: structured JSON diff for precise path-level differences
    import math

    diffs = []
    if type(a) != type(b):
        return [(path or "/", a, b)]
    if isinstance(a, dict):
        # NOTE: keys may be mixed types (int/str) across versions; sort by string representation for a comparable order
        keys = sorted(set(a.keys()) | set(b.keys()), key=lambda x: str(x))
        for k in keys:
            av = a.get(k, "<MISSING>")
            bv = b.get(k, "<MISSING>")
            diffs.extend(json_diffs(av, bv, tol, f"{path}/{k}" if path else f"/{k}"))
        return diffs
    if isinstance(a, list):
        n = max(len(a), len(b))
        for i in range(n):
            av = a[i] if i < len(a) else "<MISSING>"
            bv = b[i] if i < len(b) else "<MISSING>"
            diffs.extend(json_diffs(av, bv, tol, f"{path}[{i}]"))
        return diffs
    # NOTE: numeric tolerance for floats/ints
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            af = float(a)
            bf = float(b)
            if math.isfinite(af) and math.isfinite(bf):
                if abs(af - bf) > tol:
                    return [(path or "/", a, b)]
                return []
        except Exception:
            pass
    if a != b:
        return [(path or "/", a, b)]
    return []

def stringify_keys(o):
    # NOTE: Recursively convert dict keys to strings to avoid mixed-type key issues
    if isinstance(o, dict):
        return {str(k): stringify_keys(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [stringify_keys(v) for v in o]
    return o

def print_diffs(live_dict: dict, control_dict: dict, prec: Optional[int] = 1e-7):
    control_norm = stringify_keys(normalize(control_dict, prec))
    live_norm = stringify_keys(normalize(live_dict, prec))
    # NOTE: increase numeric tolerance to ignore small platform-dependent floating noise
    diffs = json_diffs(control_norm, live_norm, tol=1e-7)
    print(f"Found {len(diffs)} structural difference(s). Showing first 200:")
    for p, ca, la in diffs[:200]:
        try:
            # NOTE: numeric delta when both sides are numeric
            if isinstance(ca, (int, float)) and isinstance(la, (int, float)):
                delta = float(la) - float(ca)
                rel = None
                try:
                    rel = delta / float(ca) if float(ca) != 0 else None
                except Exception:
                    rel = None
                if rel is not None:
                    print(f"{p}: control={ca!r}  live={la!r}  delta={delta}  rel={rel}")
                else:
                    print(f"{p}: control={ca!r}  live={la!r}  delta={delta}")
                continue
        except Exception:
            pass
        # NOTE: for long strings/lists, print truncated preview
        def preview(x):
            try:
                s = json.dumps(x, ensure_ascii=False)
            except Exception:
                s = str(x)
            if len(s) > 200:
                return s[:200] + '...'
            return s

        print(f"{p}: control={preview(ca)!r}  live={preview(la)!r}")

    print(f"Total diffs: {len(diffs)}")

def analysis_tree_to_dict(tree):
    # NOTE: Compare live: convert the generated AnalysisTree to the same dict structure used by AnalysisTree.dump, and compare canonical json bytes
    res_dict = {}
    converted_node = tree.root._to_dict()
    res_dict[tree.root.id] = converted_node
    queue = [tree.root]
    while queue:
        parent_node = queue.pop(0)
        for child_node in parent_node.child:
            node_dict = child_node._to_dict()
            node_dict["parent"] = parent_node.id
            res_dict[child_node.id] = node_dict
            res_dict[parent_node.id]["child"].append(child_node.id)
            queue.append(child_node)
    return res_dict