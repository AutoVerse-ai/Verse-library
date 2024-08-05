# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse import BaseAgent
from verse import LaneMap
from verse.plotter.plotter2D import *
import plotly.graph_objects as go


class LineAgent(BaseAgent):
    """Dynamics of a frictionless point on a line"""

    def __init__(self, id, code=None, file_name=None):
        """Contructor for the agent
        EXACTLY one of the following should be given
        file_name: name of the decision logic (DL)
        code: pyhton string defning the decision logic (DL)
        """
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    def dynamics(self, t, state):
        """Defines the RHS of the ODE used to simulate trajectories"""
        x, vx = state
        x_dot = vx
        vx_dot = 0
        return [x_dot, vx_dot]


if __name__ == "__main__":
    aball = BallAgent(
        "red_ball", file_name="/Users/mitras/Dpp/GraphGeneration/demo/ball_bounces.py"
    )
    trace = aball.TC_simulate({"none"}, [5, 0, 2], 10, 0.05)
    fig = simulation_tree(trace, map=None, fig=go.Figure(), x_dim = 1, y_dim = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1)
    fig.show()
    print(trace)