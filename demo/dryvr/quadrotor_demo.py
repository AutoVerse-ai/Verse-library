from dryvr_agent import QuadrotorAgent 
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import * 

import plotly.graph_objects as go 
from enum import Enum, auto 

class AgentMode(Enum):
    Default = auto()

if __name__ == "__main__":
    scenario = Scenario(ScenarioConfig(parallel=False))

    quad = QuadrotorAgent('quad')
    scenario.add_agent(quad)
    # The initial position of the quadrotor is uncertain in 
    # all directions within [−0.4, 0.4] [m] and also the velocity 
    # is uncertain within [−0.4, 0.4] [m/s] for all directions
    
    # The inertial (north) position x1, the inertial (east) position x2, 
    # the altitude x3, the longitudinal velocity x4, 
    # the lateral velocity x5, the vertical velocity x6, 
    # the roll angle x7, the pitch angle x8, 
    # the yaw angle x9, the roll rate x10, 
    # the pitch rate x11, and the yaw rate x12.
    scenario.set_init(
        [
            [[-0.4,-0.4,-0.4,-0.4,-0.4,-0.4, 0, 0, 0, 0, 0, 0, 0.98],
             [ 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 1.02]]
        ],
        [
            (AgentMode.Default, )
        ]
    )
    traces = scenario.verify(5, 0.01)

    fig = go.Figure() 
    fig = reachtube_tree(traces, None, fig, 0, 3, [1,3], "lines", "trace")
    fig.show()