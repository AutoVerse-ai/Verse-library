from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map import SimpleMap3
from verse import Scenario
from enum import Enum, auto
from verse.plotter.plotter2D import *

import plotly.graph_objects as go


class VehicleMode(Enum):
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
    input_code_name = './demo/tacas2023/exp6/example_controller4.py'
    scenario = Scenario()

    scenario.add_agent(CarAgent('car1', file_name=input_code_name, initial_state=[
                       [0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]], initial_mode=(VehicleMode.Normal, TrackMode.T1)))
    scenario.add_agent(NPCAgent('car2', initial_state=[
                       [15, -0.3, 0, 0.5], [15, 0.3, 0, 0.5]], initial_mode=(VehicleMode.Normal, TrackMode.T1)))
    # scenario.add_agent(NPCAgent('car3', initial_state=[[35, -3.3, 0, 0.5], [35, -2.7, 0, 0.5]], initial_mode=(VehicleMode.Normal, TrackMode.T2)))
    # scenario.add_agent(NPCAgent('car4', initial_state=[[30, -0.5, 0, 0.5], [30, 0.5, 0, 0.5]], initial_mode=(VehicleMode.Normal, TrackMode.T1)))
    tmp_map = SimpleMap3()
    scenario.set_map(tmp_map)
    # traces = scenario.simulate(70, 0.05)
    # # traces.dump('./output1.json')
    # fig = go.Figure()
    # fig = simulation_anime(traces, tmp_map, fig, 1,
    #                        2, [1, 2], 'lines', 'trace', anime_mode='trail', full_trace = True)
    # fig.show()

    traces = scenario.verify(40, 0.05,
                             reachability_method='NeuReach',
                             params={
                                 "N_X0": 1,
                                 "N_x0": 50,
                                 "N_t": 500,
                                 "epochs": 50,
                                 "_lambda": 5,
                                 "use_cuda": True,
                                 'r': 0,
                             }
                            )
    traces.dump("./demo/tacas2023/exp6/output6_neureach.json")

    # traces = scenario.verify(40, 0.05,)
    # traces.dump("./demo/tacas2023/exp6/output6_dryvr.json")
    # traces = AnalysisTree.load('./output6.json')
    traces = scenario.verify(50, 0.05)
    fig = go.Figure()
    fig = reachtube_tree(traces, tmp_map, fig, 1,
                         2, [1, 2], 'lines', 'trace')
    fig.show()
