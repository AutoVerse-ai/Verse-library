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
from typing import Any

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

def normalize(obj: Any, float_precision: int | None = None):
    if isinstance(obj, dict):
        return {k: normalize(v, float_precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v, float_precision) for v in obj]
    if isinstance(obj, float) and float_precision is not None:
        return round(obj, float_precision)
    return obj

def canonical_json_bytes(obj: Any, float_precision: int | None = None) -> bytes:
    norm = normalize(obj, float_precision)
    s = json.dumps(norm, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return s.encode('utf-8')

def file_hash(path: str, float_precision: int | None = None) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    b = canonical_json_bytes(obj, float_precision)
    return hashlib.sha256(b).hexdigest()

def compare_files(path_a: str, path_b: str, float_precision: int | None = None) -> bool:
    return file_hash(path_a, float_precision) == file_hash(path_b, float_precision)

if __name__ == '__main__':
    import sys
    a, b = sys.argv[1], sys.argv[2]
    equal = compare_files(a, b, float_precision=None)  # set precision if desired
    print('IDENTICAL' if equal else 'DIFFER')

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
        dump_test_path = f'{script_dir}/1c1n_dump_test.json'
        
        if os.path.exists(dump_test_path):
            os.remove(dump_test_path) # NOTE: this deletion and later writing may lead to a race condition

        traces.dump(dump_test_path)
        self.assertTrue(compare_files(f'{script_dir}/1c1n_dump_test.json', f'{script_dir}/1c1n_dump_control.json'))
        # pass
        print("Highway (1c1n, straight, 3 lane) verification test Passed")
        # fig = go.Figure()
        # fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace")
        # fig = reachtube_tree_video(traces, None, fig, 1, 2, [1, 2], plot_color=colors, output_path="test.mp4", show_legend=True)
        # fig = reachtube_tree_video(traces, None, fig, 1, 2, [1, 2], plot_color=colors, show_legend=True)
        # fig.show()

if __name__ == "__main__":
    unittest.main()