from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario import Scenario, ScenarioConfig
from enum import Enum, auto
from verse.plotter.plotter2D import *

import time
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
    input_code_name = './demo/cav2023/exp8/example_controller4.py'
    scenario = Scenario()

    scenario.add_agent(CarAgent('car1', file_name=input_code_name, initial_state=[
                       [0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]], initial_mode=(AgentMode.Normal, TrackMode.T1)))
    scenario.add_agent(NPCAgent('car2', initial_state=[
                       [10, -0.3, 0, 0.5], [10, 0.3, 0, 0.5]], initial_mode=(AgentMode.Normal, TrackMode.T1)))
    scenario.add_agent(NPCAgent('car3', initial_state=[[25, 2.7, 0, 0.5], [25, 3.3, 0, 0.5]], initial_mode=(AgentMode.Normal, TrackMode.T0)))
    # scenario.add_agent(NPCAgent('car4', initial_state=[[30, -0.5, 0, 0.5], [30, 0.5, 0, 0.5]], initial_mode=(AgentMode.Normal, TrackMode.T1)))
    tmp_map = M1()
    scenario.set_map(tmp_map)
    # traces = scenario.simulate(70, 0.05)
    # # traces.dump('./output1.json')
    # fig = go.Figure()
    # fig = simulation_anime(traces, tmp_map, fig, 1,
    #                        2, [1, 2], 'lines', 'trace', anime_mode='trail', full_trace = True)
    # fig.show()

    start_time = time.time()
    traces = scenario.verify(60, 0.05)
    run_time = time.time() - start_time
    traces.dump("./demo/cav2023/exp8/output8.json")
    print({
        "#A": len(scenario.agent_dict),
        "A": "C",
        "Map": "M1",
        "postCont": "DryVR",
        "Noisy S": "No",
        "# Tr": len(traces.nodes),
        "Run Time": run_time,
    })
    # traces = AnalysisTree.load('./output6.json')
    # traces = scenario.verify(50, 0.05)
    if len(sys.argv)>1 and sys.argv[1]=='p':
        fig = go.Figure()
        fig = reachtube_tree(traces, tmp_map, fig, 1,
                            2, [1, 2], 'lines', 'trace')
        fig.show()
