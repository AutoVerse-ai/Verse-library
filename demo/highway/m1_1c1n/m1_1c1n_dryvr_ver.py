from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import  Scenario, ScenarioConfig
from enum import Enum, auto
from verse.plotter.plotter2D import *

import sys
import plotly.graph_objects as go


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


if __name__ == "__main__":
    import os

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
    fig = go.Figure()
    # fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace")
    # fig = reachtube_tree_video(traces, None, fig, 1, 2, [1, 2], plot_color=colors, output_path="test.mp4", show_legend=True)
    fig = reachtube_tree_video(traces, None, fig, 1, 2, [1, 2], plot_color=colors, show_legend=True)
    fig.show()
