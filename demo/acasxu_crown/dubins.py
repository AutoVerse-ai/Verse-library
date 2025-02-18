from dubins_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
import sys
import plotly.graph_objects as go

class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()

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
    input_code_name = os.path.join(script_dir, "controller.py")
    car = CarAgent('car1', file_name=input_code_name)
    car2 = NPCAgent('car2')
    scenario = Scenario(ScenarioConfig(parallel=False))
    car.set_initial(
                initial_state=[[0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]],
        initial_mode=(AgentMode.SR, TrackMode.T1)
    )
    car2.set_initial(
            initial_state=[[15, -0.3, 0, 0.5], [15, 0.3, 0, 0.5]],
        initial_mode=(AgentMode.COC, TrackMode.T1)
    )
    scenario.add_agent(car)
    scenario.add_agent(car2)
    trace = scenario.verify(0.1,0.1) # increasing ts to 0.1 to increase learning speed, do the same for dryvr
    fig = reachtube_tree(trace) 
    fig.show()