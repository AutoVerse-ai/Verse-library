from enum import Enum, auto
import copy
from math import pi, tan, cos, sin

# Configuration Parameters
zeta = 0.9
# coefficient of restitution
ms = 3.2
# mass of sleeve(kg)
mg2 = 18.1
# mass of second gear(kg)
Jg2 = 0.7
# inertia of second gear(kg*m**2)
ig2 = 3.704
# gear ratio of second gear
Rs = 0.08
# radius of sleeve(m)
theta = pi * (36 / 180)
# included angle of gear(rad)
b = 0.01
# width of gear spline(m)
deltap = -0.003
# axial position where sleeve engages with second gear(m)

# hard coded parameters
# cos(theta)**2 = 0.6545084971874737
# cos(theta) = 0.8090169943749475
# sin(theta)**2 = 0.3454915028125263
# sin(theta) = 0.5877852522924731
# tan(theta) = 0.7265425280053609


class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


class State:
    px = 0.0
    py = 0.0
    vx = 0.0
    vy = 0.0
    i = 0.0
    agent_mode: AgentMode = AgentMode.Free

    def __init__(self, x, y, agent_mode: AgentMode):
        pass


def forbidden(ego: State):
    res = ego.i >= 20
    return res


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.agent_mode == AgentMode.Free:
        if (ego.py >= -ego.px * 0.7265425280053609) and (
            ego.vx * 0.5877852522924731 + ego.vy * 0.8090169943749475 > 0
        ):
            output.i = ego.i + (ego.vx * 0.5877852522924731 + ego.vy * 0.8090169943749475) * (
                zeta + 1
            ) * ms * mg2 / (ms * (0.6545084971874737) + mg2 * (0.3454915028125263))
            output.vx = (
                ego.vx * (ms * 0.6545084971874737 - mg2 * zeta * 0.3454915028125263)
                + ego.vy * (-(zeta + 1) * mg2 * 0.5877852522924731 * 0.8090169943749475)
            ) / (ms * (0.6545084971874737) + mg2 * (0.3454915028125263))
            output.vy = (
                ego.vx * (-(zeta + 1) * ms * 0.5877852522924731 * 0.8090169943749475)
                + ego.vy * (mg2 * 0.3454915028125263 - ms * zeta * 0.6545084971874737)
            ) / (ms * (0.6545084971874737) + mg2 * (0.3454915028125263))
        if (ego.py <= ego.px * 0.7265425280053609) and (
            ego.vx * 0.5877852522924731 - ego.vy * 0.8090169943749475 > 0
        ):
            output.i = ego.i + (ego.vx * 0.5877852522924731 - ego.vy * 0.8090169943749475) * (
                zeta + 1
            ) * ms * mg2 / (ms * (0.6545084971874737) + mg2 * (0.3454915028125263))
            output.vx = (
                ego.vx * (ms * 0.6545084971874737 - mg2 * zeta * 0.3454915028125263)
                + ego.vy * ((zeta + 1) * mg2 * 0.5877852522924731 * 0.8090169943749475)
            ) / (ms * (0.6545084971874737) + mg2 * (0.3454915028125263))
            output.vy = (
                ego.vx * ((zeta + 1) * ms * 0.5877852522924731 * 0.8090169943749475)
                + ego.vy * (mg2 * 0.3454915028125263 - ms * zeta * 0.6545084971874737)
            ) / (ms * (0.6545084971874737) + mg2 * (0.3454915028125263))
        if ego.px >= deltap:
            output.agent_mode = AgentMode.Meshed
            if (ego.vx >= 0) and (ego.vy >= 0):
                output.i = ego.i + ms * ego.vx + ms * ego.vy
            if (ego.vx >= 0) and (ego.vy < 0):
                output.i = ego.i + ms * ego.vx - ms * ego.vy
            if (ego.vx < 0) and (ego.vy >= 0):
                output.i = ego.i - ms * ego.vx + ms * ego.vy
            if (ego.vx < 0) and (ego.vy < 0):
                output.i = ego.i - ms * ego.vx - ms * ego.vy
            output.vx = 0
            output.vy = 0

    # assert not forbidden(ego), "forbidden state"

    return output
