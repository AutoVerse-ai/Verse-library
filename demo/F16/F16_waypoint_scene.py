'''
Sayan Mitra
Derived from Stanley Bak's version
'''


from typing import Tuple, List

import numpy as np
from scipy.integrate import ode
from enum import Enum, auto

from dryvr_plus_plus.scene_verifier.agents.base_agent import BaseAgent
from dryvr_plus_plus.scene_verifier.map.lane_map import LaneMap
from dryvr_plus_plus.scene_verifier.code_parser.pythonparser import EmptyAst


import math

from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim

from aerobench.visualize import plot

from waypoint_autopilot import WaypointAutopilot


class F16Mode(Enum):
    '''Defines the discrete modes of a single agent'''
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.


class State:
    '''Defines the state variables of the model
        Both discrete and continuous variables.
        Initial values defined here do not matter.
    '''
    mode: F16Mode
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
    psi = math.pi / 8  # Yaw angle from North (rad)

    P = 0
    Q = 0
    R = 0
    pn = 0
    pe = 0

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    def __init__(self, vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, alt, power, mode:F16Mode):
        pass

def controller(ego:State, others:State):
    '''Computes the possible mode transitions
        For now this is an empty controller function.
        Coming soon. Waypoint transitions. Platooning.'''
    output = copy.deepcopy(ego)
    return output

def main():
    ''' The main function defines and simulates a scene.
        Defining and using a  scenario involves the following 5 easy steps:
        1. creating a basic scenario object with Scenario()
        2. defining the agents that will populate the object, here we have two ball agents
        3. adding the agents to the scenario using .add_agent()
        4. initializing the agents for this scenario.
            Note that agents are only initialized *in* a scenario, not individually outside a scenario
        5. genetating the simulation traces or computing the reachable states
    '''
    F16waypointScene = Scenario()
    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 3800        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = math.pi/8   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 70 # simulation time

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
    '''Main call to simulation'''
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')

    print(f"Simulation Completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    plot.plot_single(res, 'alt', title='Altitude (ft)')
    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_overhead(res, waypoints=waypoints)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_attitude(res)
    filename = 'attitude.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot inner loop controls + references
    plot.plot_inner_loop(res)
    filename = 'inner_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot outer loop controls + references
    plot.plot_outer_loop(res)
    filename = 'outer_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    main()
