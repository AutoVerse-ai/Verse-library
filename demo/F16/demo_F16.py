
from verse import Scenario
from verse.plotter.plotter2D import *
from agent_F16 import F16_Agent
import plotly.graph_objects as go
from enum import Enum, auto
from numpy import deg2rad
from math import pi
from waypoint_autopilot import WaypointAutopilot
import os


class F16Mode(Enum):
    '''Defines the discrete modes of a single agent'''
    NORMAL = auto()
    # TODO: The one mode of this automation is called "NORMAL" and auto assigns it an integer value.


class State:
    '''Defines the state variables of the model
        Both discrete and continuous variables.
        Initial values defined here do not matter.
    '''
    mode: F16Mode = F16Mode.NORMAL
    ### Continuous variable initial conditions ###
    power = 9  # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)

    # Initial Attitude
    alt = 3800  # altitude (ft)
    vt = 540  # initial velocity (ft/sec)
    phi = 0  # Roll angle from wings level (rad)
    theta = 0  # Pitch angle from nose level (rad)
    psi = pi / 8  # Yaw angle from North (rad)

    p = 0
    q = 0
    r = 0
    pn = 0
    pe = 0

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    # init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    def __init__(self, vt, alpha, beta, phi, theta, psi, p, q, r, pn, pe, alt, power, mode: F16Mode):
        pass


if __name__ == "__main__":
    path = os.path.abspath(__file__)
    input_code_name = path.replace('demo_F16.py', 'controller_F16.py')
    # input_code_name = './F16/controller_F16.py'
    scenario = Scenario()

    ### Initial Conditions ###
    power = 9  # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 3800        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = pi/8   # Yaw angle from North (rad)

    # make waypoint list
    e_pt = 1000
    n_pt = 3000
    h_pt = 4000

    waypoints = [[e_pt, n_pt, h_pt],
                 [e_pt + 2000, n_pt + 5000, h_pt - 100],
                 [e_pt - 2000, n_pt + 15000, h_pt - 250],
                 [e_pt - 500, n_pt + 25000, h_pt]]

    ap = WaypointAutopilot(waypoints, stdout=True)
    step = 1/30
    extended_states = True
    test = F16_Agent('test', ap=ap, extended_states=extended_states,
                     integrator_str='rk45', file_name=input_code_name)
    scenario.add_agent(test)
    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, p, q, r, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 70  # simulation time
    scenario.set_init(
        [
            [init, init],
        ],
        [
            tuple([F16Mode.NORMAL]),
        ]
    )
    traces = scenario.simulate(tmax, step)
    fig = go.Figure()
    fig = simulation_tree(traces, None, fig, 0, 12,
                          'lines', 'trace', print_dim_list=[1, 2], label_mode='None')
    fig.show()
