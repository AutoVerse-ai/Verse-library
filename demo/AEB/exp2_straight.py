from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse import Scenario
from verse.scenario import ScenarioConfig
# from noisy_sensor import NoisyVehicleSensor
from verse.plotter.plotter2D import *

from enum import Enum, auto
import time
import sys
import plotly.graph_objects as go

class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()        # Pedestrians
    Sign = auto()       # Signs, stop signs, merge, yield etc.
    Signal = auto()     # Traffic lights
    Obstacle = auto()   # Static (to road/lane) obstacles


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
    input_code_name = './demo/AEB/controller1.py'
    scenario = Scenario(ScenarioConfig(init_seg_length=5))

    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = NPCAgent('car2')
    scenario.add_agent(car)
    # car = NPCAgent('car3')
    # scenario.add_agent(car)
    # Q. Why do we need the tmp_map name? 
    tmp_map = M1()
    scenario.set_map(M1())
    scenario.set_init(
        [
            [[5, -0.5, 0, 1.0], [5.5, 0.5, 0, 1.0]],
            [[20, -0.2, 0, 0.5], [20, 0.2, 0, 0.5]],
#            [[4-2.5, 2.8, 0, 1.0], [4.5-2.5, 3.2, 0, 1.0]],
        ],
        [
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T1),
#            (AgentMode.Normal, TrackMode.T0),
        ]
    )

    start_time = time.time()
    # traces = scenario.verify(40, 0.1, params={"bloating_method": 'GLOBAL'})
    traces = scenario.simulate(100,0.1)
    run_time = time.time()-start_time 
    traces.dump('./demo/AEB/sim_straight.json')

    print({
        "#A": len(scenario.agent_dict),
        "A": "C",
        "Map": "M1",
        "postCont": "DryVR",
        "Noisy S": "No",
        "# Tr": len(traces.nodes),
        "Run Time": run_time,
    })

    fig = go.Figure()
    fig = simulation_tree(traces, tmp_map, fig, 1, 2,None, 'lines', 'trace')
    # fig = simulation_anime(traces, tmp_map, fig, 1, 2,None, 'lines', 'trace', time_step=0.1)
    # fig = reachtube_anime(traces, tmp_map, fig, 1, 2, None,'lines', 'trace', combine_rect=1)
    fig.show()