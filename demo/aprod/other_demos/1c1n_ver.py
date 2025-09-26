from car_agent import CarAgent, NPCAgent
from car_sensor import CarSensor
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from verse.analysis.verifier import ReachabilityMethod
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse import Scenario, ScenarioConfig

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

class GPSMode(Enum):
    Passive = auto()
    Active = auto()

if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "controller.py")
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    # car_sensor = CarSensor()
    # scenario.set_sensor(car_sensor)
    base_l, base_u = [0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]
    init_err = [0.1, 0.1, 0, 0.01]
    no_err = [0, 0,0,0]
    scenario.add_agent(
        CarAgent(
            "car1",
            file_name=input_code_name, 
            # structure is x,hx,ex,t
            initial_state=[[0, -0.5, 0, 1.0]+ [base_l[i]-init_err[i] for i in range(4)] + [-0.1, -0.1, 0, -0.01] + [0],  
                           [0.01, 0.5, 0, 1.0]+[base_u[i]+init_err[i] for i in range(4)]+ init_err + [0]],
            initial_mode=(AgentMode.Normal, TrackMode.T1, GPSMode.Passive),
        )
    )
    scenario.add_agent(
        NPCAgent(
            "car2",
            initial_state=[[15, -0.3, 0, 0.5], [15, 0.3, 0, 0.5]],
            initial_mode=(AgentMode.Normal, TrackMode.T1),
        )
    )
    tmp_map = M1()
    scenario.set_map(tmp_map)
    time_step = 0.05

    traces = scenario.verify(40, time_step)
    fig = go.Figure()
    fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace")
    fig.show()
