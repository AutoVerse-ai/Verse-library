from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import  Scenario, ScenarioConfig
from enum import Enum, auto
from verse.plotter.plotter2D import *
# from compare_json import compare_files

import sys
import plotly.graph_objects as go
import os
import unittest
import json
import hashlib
from typing import Any, Optional

class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
    M21 = auto()
    M10 = auto()

def normalize(obj: Any, float_precision: Optional[int] = None):
    if isinstance(obj, dict):
        return {k: normalize(v, float_precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v, float_precision) for v in obj]
    if isinstance(obj, float) and float_precision is not None:
        return round(obj, float_precision)
    return obj

def canonical_json_bytes(obj: Any, float_precision: Optional[int] = None) -> bytes:
    norm = normalize(obj, float_precision)
    s = json.dumps(norm, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return s.encode('utf-8')

class TestVerify(unittest.TestCase):

    def test_highway_1c1n(self):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "example_controller4.py")
        scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
        scenario.add_agent(
            CarAgent(
                "car1",
                file_name=input_code_name,
                initial_state=[[0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]],
                initial_mode=(AgentMode.Normal, TrackMode.T1),
            )
        )
        scenario.add_agent(
            NPCAgent(
                "car2",
                initial_state=[[15, -0.3, 0, 0.5], [15, 0.3, 0, 0.5]],
                initial_mode=(AgentMode.Normal, TrackMode.T1),
            )
        )
        # scenario.add_agent(NPCAgent('car3', initial_state=[[35, -3.3, 0, 0.5], [35, -2.7, 0, 0.5]], initial_mode=(AgentMode.Normal, TrackMode.T2)))
        # scenario.add_agent(NPCAgent('car4', initial_state=[[30, -0.5, 0, 0.5], [30, 0.5, 0, 0.5]], initial_mode=(AgentMode.Normal, TrackMode.T1)))
        tmp_map = M1()
        scenario.set_map(tmp_map)
        time_step = 0.05

        traces = scenario.verify(40, time_step)

        # NOTE: Compare live: convert the generated AnalysisTree to the same dict structure used by AnalysisTree.dump, and compare canonical json bytes
        def analysis_tree_to_dict(tree):
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

        live_dict = analysis_tree_to_dict(traces)
        control_path = os.path.join(script_dir, '1c1n_dump_control.json')
        with open(control_path, 'r', encoding='utf-8') as f:
            control_dict = json.load(f)

        prec = 10
        live_text = canonical_json_bytes(live_dict, float_precision=prec).decode('utf-8')
        control_text = canonical_json_bytes(control_dict, float_precision=prec).decode('utf-8')
        live_hash = hashlib.sha256(live_text.encode('utf-8')).hexdigest()
        control_hash = hashlib.sha256(control_text.encode('utf-8')).hexdigest()

        if live_hash != control_hash:
            # NOTE: structured JSON diff for precise path-level differences
            def json_diffs(a, b, tol=0.0, path=""):
                import math

                diffs = []
                if type(a) != type(b):
                    return [(path or "/", a, b)]
                if isinstance(a, dict):
                    keys = sorted(set(a.keys()) | set(b.keys()))
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

            # normalize with the same precision used for canonical bytes
            control_norm = normalize(control_dict, prec)
            live_norm = normalize(live_dict, prec)
            diffs = json_diffs(control_norm, live_norm, tol=1e-10)
            print(f"Found {len(diffs)} structural difference(s). Showing first 200:")
            for p, ca, la in diffs[:200]:
                try:
                    # numeric delta when both sides are numeric
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
                # for long strings/lists, print truncated preview
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
        self.assertEqual(live_hash, control_hash)
        print("Highway (1c1n, straight, 3 lane) verification test Passed")

if __name__ == "__main__":
    unittest.main()