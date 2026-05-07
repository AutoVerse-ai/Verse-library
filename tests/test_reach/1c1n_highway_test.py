from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import  Scenario, ScenarioConfig
from enum import Enum, auto
from verse.plotter.plotter2D import *
from tests.test_utils import analysis_tree_to_dict, canonical_json_bytes, print_diffs

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

        live_dict = analysis_tree_to_dict(traces) # NOTE: has some structure as traces.dump() except in dict instead of stored file
        control_path = os.path.join(script_dir, '1c1n_dump_control.json')
        with open(control_path, 'r', encoding='utf-8') as f:
            control_dict = json.load(f)

        # NOTE: reduce precision to avoid spurious diffs from sub-nanosecond noise
        prec = 7
        live_text = canonical_json_bytes(live_dict, float_precision=prec).decode('utf-8')
        control_text = canonical_json_bytes(control_dict, float_precision=prec).decode('utf-8')
        live_hash = hashlib.sha256(live_text.encode('utf-8')).hexdigest()
        control_hash = hashlib.sha256(control_text.encode('utf-8')).hexdigest()

        if live_hash != control_hash:
            print_diffs(live_dict, control_dict)

        self.assertEqual(live_hash, control_hash)
        print("Highway (1c1n, straight, 3 lane) verification test Passed")

if __name__ == "__main__":
    unittest.main()