from dubins_3d_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
import sys
import plotly.graph_objects as go
import torch
# from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
import pyvista as pv

from dubin_sensor_3D import DubinSensor

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

def get_acas_state(own_state: List[float], int_state: List[float]) -> torch.Tensor:
    dist = np.sqrt((own_state[0]-int_state[0])**2+(own_state[1]-int_state[1])**2)
    theta = wrap_to_pi((2*np.pi-own_state[2])+np.arctan2(int_state[0], int_state[1]))
    psi = wrap_to_pi(int_state[2]-own_state[2])
    return torch.tensor([dist, theta, own_state[3], int_state[3]])


if __name__ == "__main__":
    import os
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "controller_v2_3D.py")
    car = CarAgent('car1', file_name=input_code_name)
    car2 = NPCAgent('car2')
    scenario = Scenario(ScenarioConfig(parallel=False))
    scenario.config.print_level = 0
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    scenario.set_sensor(DubinSensor())
    car.set_initial(
                initial_state=[[-1, -1010, -1, np.pi/3, np.pi/6, 100, 0], [1, -990, 1, np.pi/3, np.pi/6, 100, 0]],
        initial_mode=(AgentMode.COC, TrackMode.T1)
    )
    car2.set_initial(
            initial_state=[[-2001, -10, 999, 0,0, 100, 0], [-1999, 10, 1001, 0,0, 100, 0]],
        initial_mode=(AgentMode.COC, TrackMode.T1)
    )
    scenario.add_agent(car)
    scenario.add_agent(car2)
    #trace = scenario.simulate(4, 1)
    plotter = pv.Plotter()
    trace = scenario.verify(20, 1, plotter) # increasing ts to 0.1 to increase learning speed, do the same for dryvr2
    fig = go.Figure()
    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    # fig = reachtube_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig = reachtube_tree_3d(trace, fig,1,'x', 2,'y',3,'z')
    fig.show()