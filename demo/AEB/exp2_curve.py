from verse.agents.example_agent import CarAgent, NPCAgent

# from verse.map.example_map import SimpleMap3, SimpleMap6
from verse.map.example_map.map_tacas import M3
from verse import Scenario
from verse.scenario import ScenarioConfig

# from noisy_sensor import NoisyVehicleSensor
from verse.plotter.plotter2D import *

# from verse.plotter.plotter2D_old import plot_reachtube_tree, plot_map

from enum import Enum, auto
import time
import sys
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import pyvista as pv
from verse.plotter.plotter3D import *


class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


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


class State:
    x: float
    y: float
    theta: float
    v: float
    agent_mode: AgentMode
    track_mode: TrackMode

    def __init__(self, x, y, theta, v, agent_mode: AgentMode, track_mode: TrackMode):
        pass


if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "example_controller5.py")
 
    scenario = Scenario(ScenarioConfig(init_seg_length=5, parallel=False))

    car = CarAgent("car1", file_name=input_code_name)
    scenario.add_agent(car)
    car = NPCAgent("car2")
    scenario.add_agent(car)
    car = NPCAgent("car3")
    scenario.add_agent(car)
    tmp_map = M3()
    scenario.set_map(tmp_map)
    scenario.set_init(
        [
            [[5, -0.5, 0, 1.0], [5.5, 0.5, 0, 1.0]],
            [[20, -0.2, 0, 0.5], [20, 0.2, 0, 0.5]],
            [[4 - 2.5, 2.8, 0, 1.0], [4.5 - 2.5, 3.2, 0, 1.0]],
        ],
        [
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T0),
        ],
    )

    start_time = time.time()
    traces = scenario.verify(40, 0.1, params={"bloating_method": "GLOBAL"})
    run_time = time.time() - start_time
    traces.dump(os.path.join(script_dir, "output2_curve.json"))  
    print(
        {
            "#A": len(scenario.agent_dict),
            "A": "C",
            "Map": "M3",
            "postCont": "DryVR",
            "Noisy S": "No",
            "# Tr": len(traces.nodes),
            "Run Time": run_time,
        }
    )

    if len(sys.argv) > 1 and sys.argv[1] == "p":
        fig = go.Figure()
        fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace")
        fig.show()

    # fig = go.Figure()
    # fig = reachtube_anime(traces, tmp_map, fig, 1,
    #                       2, 'lines', 'trace', combine_rect=1)
    # fig.show()

    # fig = pv.Plotter()
    # fig = plot3dReachtube(traces,'car1',1,2,0,'b',fig)
    # fig = plot3dReachtube(traces,'car2',1,2,0,'r',fig)
    # fig = plot3dReachtube(traces,'car3',1,2,0,'g',fig)
    # fig = plot_line_3d([0,0,0],[10,0,0],ax=fig,color='r')
    # fig = plot_line_3d([0,0,0],[0,10,0],ax=fig,color='g')
    # fig = plot_line_3d([0,0,0],[0,0,10],ax=fig,color='b')
    # fig.show()
