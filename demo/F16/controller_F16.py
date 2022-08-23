from enum import Enum, auto
import copy
from numpy import deg2rad
from math import pi


class F16Mode(Enum):
    '''Defines the discrete modes of a single agent'''
    NORMAL = auto()
    # TODO: The one mode of this automation is called "NORMAL" and auto assigns it an integer value.


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


def controller(ego: State, others: State):
    '''Computes the possible mode transitions
        For now this is an empty controller function.
        Coming soon. Waypoint transitions. Platooning.'''
    output = copy.deepcopy(ego)
    return output
