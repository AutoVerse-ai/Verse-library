from enum import Enum, auto
import copy
from typing import List


class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


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


class State:
    '''x: float
    y: float
    theta: float
    v: float
    agent_mode: AgentMode'''
    rho: float
    theta: float
    psi: float
    v_own: float
    v_int: float
    timer: float
    agent_mode: AgentMode

    def __init__(self, rho, theta, psi, v_own, v_int, agent_mode: AgentMode):#__init__(self, x, y, theta, v, agent_mode: AgentMode):
        pass


def decisionLogic(ego: State, others: List[State]):
    output = copy.deepcopy(ego)
    # assert not vehicle_close(ego, others)
    #if ego.timer >= 2.1:
    #    output.timer = 0
    if ego.agent_mode!=AgentMode.SL:
        output.agent_mode = AgentMode.SL
    #if ego.rho >= 1000:
    #    output.agent_mode = AgentMode.WR
    return output
'''
def decisionLogic(ego: State, others: List[State]):
    next = copy.deepcopy(ego)
    rho = ego.rho
    theta = ego.theta
    psi = ego.psi
    
    acas_update_time = 3.9
    
    # Variation that takes 6 min to run
    if ego.timer >= acas_update_time:
        if ego.agent_mode == AgentMode.COC:  # advisory 0
            if rho <= 15545.62:
                if theta <= -0.13:
                    if rho <= 1727.29:
                        if psi <= 0.13:
                            if rho <= 1320.87:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                            if rho > 1320.87:
                                if theta <= -1.78:
                                    next.agent_mode = AgentMode.WL
                                    next.timer = 0
                                if theta > -1.78:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                        if psi > 0.13:
                            if rho <= 711.24:
                                next.agent_mode = AgentMode.SR
                                next.timer = 0
                            if rho > 711.24:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                    if rho > 1727.29:
                        if theta <= -0.83:
                            if psi <= -0.25:
                                if rho <= 5385.08:
                                    next.agent_mode = AgentMode.WL
                                    next.timer = 0
                                if rho > 5385.08:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                            if psi > -0.25:
                                if theta <= -2.73:
                                    if psi <= 0.38:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                                    if psi > 0.38:
                                        next.agent_mode = AgentMode.WR
                                        next.timer = 0
                                if theta > -2.73:
                                    if rho <= 6197.93:
                                        if psi <= 1.9:
                                            if theta <= -2.03:
                                                next.agent_mode = AgentMode.WL
                                                next.timer = 0
                                            if theta > -2.03:
                                                if rho <= 4165.82:
                                                    next.agent_mode = AgentMode.SL
                                                    next.timer = 0
                                                if rho > 4165.82:
                                                    if psi <= 0.7:
                                                        next.agent_mode = AgentMode.WL
                                                        next.timer = 0
                                                    if psi > 0.7:
                                                        next.agent_mode = AgentMode.SL
                                                        next.timer = 0
                                        if psi > 1.9:
                                            next.agent_mode = AgentMode.WL
                                            next.timer = 0
                                    if rho > 6197.93:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                        if theta > -0.83:
                            if rho <= 4572.24:
                                if psi <= 0.76:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                                if psi > 0.76:
                                    if theta <= -0.44:
                                        next.agent_mode = AgentMode.SL
                                        next.timer = 0
                                    if theta > -0.44:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                            if rho > 4572.24:
                                if theta <= -0.38:
                                    if rho <= 9449.3:
                                        if psi <= 1.08:
                                            next.agent_mode = AgentMode.WL
                                            next.timer = 0
                                        if psi > 1.08:
                                            next.agent_mode = AgentMode.SL
                                            next.timer = 0
                                    if rho > 9449.3:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                                if theta > -0.38:
                                    if psi <= 0.83:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                                    if psi > 0.83:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                if theta > -0.13:
                    if rho <= 1727.29:
                        if psi <= -0.19:
                            if rho <= 508.03:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                            if rho > 508.03:
                                if theta <= 0.44:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                                if theta > 0.44:
                                    if theta <= 2.73:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                                    if theta > 2.73:
                                        next.agent_mode = AgentMode.SL
                                        next.timer = 0
                        if psi > -0.19:
                            next.agent_mode = AgentMode.SR
                            next.timer = 0
                    if rho > 1727.29:
                        if theta <= 1.33:
                            if rho <= 5181.87:
                                if psi <= -0.32:
                                    if theta <= 0.44:
                                        next.agent_mode = AgentMode.SL
                                        next.timer = 0
                                    if theta > 0.44:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                                if psi > -0.32:
                                    if theta <= 0.57:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                                    if theta > 0.57:
                                        if rho <= 3556.19:
                                            next.agent_mode = AgentMode.SR
                                            next.timer = 0
                                        if rho > 3556.19:
                                            next.agent_mode = AgentMode.WR
                                            next.timer = 0
                            if rho > 5181.87:
                                if psi <= -1.08:
                                    if theta <= 0.38:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                                    if theta > 0.38:
                                        if rho <= 8026.82:
                                            next.agent_mode = AgentMode.SR
                                            next.timer = 0
                                        if rho > 8026.82:
                                            next.agent_mode = AgentMode.WR
                                            next.timer = 0
                                if psi > -1.08:
                                    next.agent_mode = AgentMode.WR
                                    next.timer = 0
                        if theta > 1.33:
                            if rho <= 8636.46:
                                if psi <= 0.38:
                                    next.agent_mode = AgentMode.WR
                                    next.timer = 0
                                if psi > 0.38:
                                    if rho <= 4369.03:
                                        next.agent_mode = AgentMode.WR
                                        next.timer = 0
                                    if rho > 4369.03:
                                        next.agent_mode = AgentMode.COC
                                        next.timer = 0
                            if rho > 8636.46:
                                next.agent_mode = AgentMode.COC
                                next.timer = 0
            if rho > 15545.62:
                if rho <= 18593.78:
                    if psi <= 2.03:
                        next.agent_mode = AgentMode.COC
                        next.timer = 0
                    if psi > 2.03:
                        next.agent_mode = AgentMode.WL
                        next.timer = 0
                if rho > 18593.78:
                    next.agent_mode = AgentMode.COC
                    next.timer = 0

        
        if ego.agent_mode == AgentMode.WL: # advisory 1
            if psi <= -2.79:
                if theta <= 0.32:
                    if rho <= 13513.51:
                        if theta <= 0.06:
                            next.agent_mode = AgentMode.SL
                            next.timer = 0
                        if theta > 0.06:
                            next.agent_mode = AgentMode.SR
                            next.timer = 0
                    if rho > 13513.51:
                        if theta <= -0.32:
                            next.agent_mode = AgentMode.COC
                            next.timer = 0
                        if theta > -0.32:
                            next.agent_mode = AgentMode.WL
                            next.timer = 0
                if theta > 0.32:
                    if rho <= 10058.93:
                        if theta <= 1.27:
                            next.agent_mode = AgentMode.SR
                            next.timer = 0
                        if theta > 1.27:
                            next.agent_mode = AgentMode.SL
                            next.timer = 0
                    if rho > 10058.93:
                        if rho <= 10871.77:
                            next.agent_mode = AgentMode.WR
                            next.timer = 0
                        if rho > 10871.77:
                            next.agent_mode = AgentMode.COC
                            next.timer = 0
            if psi > -2.79:
                if rho <= 9042.88:
                    if theta <= -0.51:
                        if rho <= 1930.5:
                            if psi <= 0.13:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                            if psi > 0.13:
                                if psi <= 1.46:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                                if psi > 1.46:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                        if rho > 1930.5:
                            if psi <= -0.57:
                                next.agent_mode = AgentMode.WL
                                next.timer = 0
                            if psi > -0.57:
                                if theta <= -1.33:
                                    if rho <= 3556.19:
                                        if psi <= 1.78:
                                            next.agent_mode = AgentMode.SL
                                            next.timer = 0
                                        if psi > 1.78:
                                            next.agent_mode = AgentMode.WL
                                            next.timer = 0
                                    if rho > 3556.19:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                                if theta > -1.33:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                    if theta > -0.51:
                        if psi <= -0.25:
                            if theta <= 0.38:
                                if rho <= 4165.82:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                                if rho > 4165.82:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                            if theta > 0.38:
                                if rho <= 1524.08:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                                if rho > 1524.08:
                                    if theta <= 2.03:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                                    if theta > 2.03:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                        if psi > -0.25:
                            if theta <= 1.21:
                                if theta <= -0.19:
                                    if psi <= 2.54:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                                    if psi > 2.54:
                                        next.agent_mode = AgentMode.SL
                                        next.timer = 0
                                if theta > -0.19:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                            if theta > 1.21:
                                if rho <= 1930.5:
                                    if psi <= 0.44:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                                    if psi > 0.44:
                                        next.agent_mode = AgentMode.SL
                                        next.timer = 0
                                if rho > 1930.5:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                if rho > 9042.88:
                    if theta <= 0.76:
                        if theta <= -1.08:
                            if rho <= 13513.51:
                                next.agent_mode = AgentMode.COC
                                next.timer = 0
                            if rho > 13513.51:
                                next.agent_mode = AgentMode.COC
                                next.timer = 0
                        if theta > -1.08:
                            if psi <= 0.57:
                                if psi <= -1.33:
                                    if theta <= -0.19:
                                        next.agent_mode = AgentMode.COC
                                        next.timer = 0
                                    if theta > -0.19:
                                        if rho <= 13107.09:
                                            next.agent_mode = AgentMode.SL
                                            next.timer = 0
                                        if rho > 13107.09:
                                            next.agent_mode = AgentMode.WL
                                            next.timer = 0
                                if psi > -1.33:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                            if psi > 0.57:
                                if theta <= 0.06:
                                    next.agent_mode = AgentMode.WL
                                    next.timer = 0
                                if theta > 0.06:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                    if theta > 0.76:
                        next.agent_mode = AgentMode.COC
                        next.timer = 0

            
        if ego.agent_mode == AgentMode.WR: # advisory 2
            if rho <= 7213.98:
                if theta <= 0.44:
                    if theta <= -2.22:
                        if rho <= 1524.08:
                            next.agent_mode = AgentMode.SR
                            next.timer = 0
                        if rho > 1524.08:
                            next.agent_mode = AgentMode.WR
                            next.timer = 0
                    if theta > -2.22:
                        if theta <= -0.63:
                            if rho <= 1117.66:
                                next.agent_mode = AgentMode.SR
                                next.timer = 0
                            if rho > 1117.66:
                                if psi <= -0.13:
                                    next.agent_mode = AgentMode.WR
                                    next.timer = 0
                                if psi > -0.13:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                        if theta > -0.63:
                            if psi <= -0.19:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                            if psi > -0.19:
                                if theta <= -0.32:
                                    if psi <= 2.03:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                                    if psi > 2.03:
                                        next.agent_mode = AgentMode.SL
                                        next.timer = 0
                                if theta > -0.32:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                if theta > 0.44:
                    if rho <= 1727.29:
                        if psi <= -0.19:
                            if psi <= -1.84:
                                next.agent_mode = AgentMode.SR
                                next.timer = 0
                            if psi > -1.84:
                                if theta <= 0.83:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                                if theta > 0.83:
                                    if theta <= 2.48:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                                    if theta > 2.48:
                                        next.agent_mode = AgentMode.SL
                                        next.timer = 0
                        if psi > -0.19:
                            next.agent_mode = AgentMode.SR
                            next.timer = 0
                    if rho > 1727.29:
                        if theta <= 1.52:
                            if psi <= 0.0:
                                next.agent_mode = AgentMode.SR
                                next.timer = 0
                            if psi > 0.0:
                                next.agent_mode = AgentMode.WR
                                next.timer = 0
                        if theta > 1.52:
                            next.agent_mode = AgentMode.WR
                            next.timer = 0
            if rho > 7213.98:
                if rho <= 8839.67:
                    if theta <= -0.51:
                        if theta <= -0.83:
                            next.agent_mode = AgentMode.COC
                            next.timer = 0
                        if theta > -0.83:
                            next.agent_mode = AgentMode.WL
                            next.timer = 0
                    if theta > -0.51:
                        if theta <= 1.02:
                            next.agent_mode = AgentMode.SR
                            next.timer = 0
                        if theta > 1.02:
                            next.agent_mode = AgentMode.WR
                            next.timer = 0
                if rho > 8839.67:
                    if theta <= -1.02:
                        next.agent_mode = AgentMode.COC
                        next.timer = 0
                    if theta > -1.02:
                        if theta <= 0.95:
                            if rho <= 12294.25:
                                next.agent_mode = AgentMode.WR
                                next.timer = 0
                            if rho > 12294.25:
                                if psi <= 0.89:
                                    if psi <= -1.59:
                                        if theta <= -0.13:
                                            next.agent_mode = AgentMode.COC
                                            next.timer = 0
                                        if theta > -0.13:
                                            next.agent_mode = AgentMode.WR
                                            next.timer = 0
                                    if psi > -1.59:
                                        next.agent_mode = AgentMode.COC
                                        next.timer = 0
                                if psi > 0.89:
                                    if theta <= 0.19:
                                        next.agent_mode = AgentMode.WR
                                        next.timer = 0
                                    if theta > 0.19:
                                        next.agent_mode = AgentMode.COC
                                        next.timer = 0
                        if theta > 0.95:
                            if rho <= 13919.93:
                                if psi <= -0.19:
                                    next.agent_mode = AgentMode.WR
                                    next.timer = 0
                                if psi > -0.19:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                            if rho > 13919.93:
                                next.agent_mode = AgentMode.COC
                                next.timer = 0

        if ego.agent_mode == AgentMode.SL: # advisory 3
            if rho <= 8839.67:
                if rho <= 4775.45:
                    if theta <= -0.44:
                        if psi <= 0.0:
                            next.agent_mode = AgentMode.SL
                            next.timer = 0
                        if psi > 0.0:
                            if rho <= 914.45:
                                if psi <= 1.4:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                                if psi > 1.4:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                            if rho > 914.45:
                                if theta <= -0.63:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                                if theta > -0.63:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                    if theta > -0.44:
                        if rho <= 711.24:
                            if psi <= -0.19:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                            if psi > -0.19:
                                if theta <= 1.02:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                                if theta > 1.02:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                        if rho > 711.24:
                            if theta <= 0.13:
                                if psi <= -0.06:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                                if psi > -0.06:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                            if theta > 0.13:
                                if theta <= 2.35:
                                    if theta <= 0.51:
                                        if psi <= -0.44:
                                            if psi <= -2.41:
                                                next.agent_mode = AgentMode.SR
                                                next.timer = 0
                                            if psi > -2.41:
                                                next.agent_mode = AgentMode.SL
                                                next.timer = 0
                                        if psi > -0.44:
                                            next.agent_mode = AgentMode.SR
                                            next.timer = 0
                                    if theta > 0.51:
                                        next.agent_mode = AgentMode.SR
                                        next.timer = 0
                                if theta > 2.35:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                if rho > 4775.45:
                    if theta <= 0.95:
                        if theta <= -0.32:
                            if theta <= -1.9:
                                next.agent_mode = AgentMode.COC
                                next.timer = 0
                            if theta > -1.9:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                        if theta > -0.32:
                            if psi <= 1.65:
                                if theta <= 0.38:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                                if theta > 0.38:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                            if psi > 1.65:
                                next.agent_mode = AgentMode.SR
                                next.timer = 0
                    if theta > 0.95:
                        if psi <= -0.89:
                            next.agent_mode = AgentMode.SR
                            next.timer = 0
                        if psi > -0.89:
                            if theta <= 1.4:
                                next.agent_mode = AgentMode.WR
                                next.timer = 0
                            if theta > 1.4:
                                next.agent_mode = AgentMode.COC
                                next.timer = 0
            if rho > 8839.67:
                if theta <= -1.4:
                    next.agent_mode = AgentMode.COC
                    next.timer = 0
                if theta > -1.4:
                    if theta <= 1.14:
                        if rho <= 13919.93:
                            if psi <= -1.59:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                            if psi > -1.59:
                                if psi <= 2.09:
                                    next.agent_mode = AgentMode.WL
                                    next.timer = 0
                                if psi > 2.09:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                        if rho > 13919.93:
                            if rho <= 35257.06:
                                if psi <= 0.51:
                                    if theta <= -0.19:
                                        if psi <= -1.27:
                                            next.agent_mode = AgentMode.COC
                                            next.timer = 0
                                        if psi > -1.27:
                                            next.agent_mode = AgentMode.SL
                                            next.timer = 0
                                    if theta > -0.19:
                                        if psi <= -0.32:
                                            next.agent_mode = AgentMode.WL
                                            next.timer = 0
                                        if psi > -0.32:
                                            next.agent_mode = AgentMode.COC
                                            next.timer = 0
                                if psi > 0.51:
                                    if theta <= 0.25:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                                    if theta > 0.25:
                                        next.agent_mode = AgentMode.COC
                                        next.timer = 0
                            if rho > 35257.06:
                                if theta <= -1.02:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                                if theta > -1.02:
                                    if psi <= 1.21:
                                        if psi <= -0.7:
                                            if theta <= -0.19:
                                                next.agent_mode = AgentMode.COC
                                                next.timer = 0
                                            if theta > -0.19:
                                                next.agent_mode = AgentMode.WL
                                                next.timer = 0
                                        if psi > -0.7:
                                            if theta <= -0.76:
                                                next.agent_mode = AgentMode.WL
                                                next.timer = 0
                                            if theta > -0.76:
                                                next.agent_mode = AgentMode.COC
                                                next.timer = 0
                                    if psi > 1.21:
                                        if theta <= 0.19:
                                            next.agent_mode = AgentMode.WL
                                            next.timer = 0
                                        if theta > 0.19:
                                            next.agent_mode = AgentMode.COC
                                            next.timer = 0
                    if theta > 1.14:
                        if theta <= 1.9:
                            if rho <= 39727.69:
                                next.agent_mode = AgentMode.COC
                                next.timer = 0
                            if rho > 39727.69:
                                if psi <= 0.06:
                                    if psi <= -0.95:
                                        next.agent_mode = AgentMode.COC
                                        next.timer = 0
                                    if psi > -0.95:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                                if psi > 0.06:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                        if theta > 1.9:
                            next.agent_mode = AgentMode.COC
                            next.timer = 0

            
        if ego.agent_mode == AgentMode.SR: # advisory 4
            if rho <= 11074.98:
                if theta <= -1.08:
                    if rho <= 4978.66:
                        if rho <= 1524.08:
                            if psi <= 0.51:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                            if psi > 0.51:
                                next.agent_mode = AgentMode.SR
                                next.timer = 0
                        if rho > 1524.08:
                            if psi <= -0.19:
                                next.agent_mode = AgentMode.COC
                                next.timer = 0
                            if psi > -0.19:
                                if psi <= 1.52:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                                if psi > 1.52:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                    if rho > 4978.66:
                        if theta <= -1.84:
                            next.agent_mode = AgentMode.COC
                            next.timer = 0
                        if theta > -1.84:
                            if psi <= 1.14:
                                if psi <= 0.19:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                                if psi > 0.19:
                                    if rho <= 5791.51:
                                        if psi <= 0.32:
                                            next.agent_mode = AgentMode.WL
                                            next.timer = 0
                                        if psi > 0.32:
                                            next.agent_mode = AgentMode.SL
                                            next.timer = 0
                                    if rho > 5791.51:
                                        next.agent_mode = AgentMode.WL
                                        next.timer = 0
                            if psi > 1.14:
                                next.agent_mode = AgentMode.SL
                                next.timer = 0
                if theta > -1.08:
                    if theta <= 0.51:
                        if psi <= -0.06:
                            next.agent_mode = AgentMode.SL
                            next.timer = 0
                        if psi > -0.06:
                            if theta <= -0.25:
                                if rho <= 1524.08:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                                if rho > 1524.08:
                                    if theta <= -1.02:
                                        if psi <= 0.57:
                                            next.agent_mode = AgentMode.WL
                                            next.timer = 0
                                        if psi > 0.57:
                                            next.agent_mode = AgentMode.SL
                                            next.timer = 0
                                    if theta > -1.02:
                                        if theta <= -0.57:
                                            next.agent_mode = AgentMode.SL
                                            next.timer = 0
                                        if theta > -0.57:
                                            if psi <= 2.28:
                                                next.agent_mode = AgentMode.SR
                                                next.timer = 0
                                            if psi > 2.28:
                                                next.agent_mode = AgentMode.SL
                                                next.timer = 0
                            if theta > -0.25:
                                next.agent_mode = AgentMode.SR
                                next.timer = 0
                    if theta > 0.51:
                        if rho <= 2743.34:
                            if psi <= -0.32:
                                if psi <= -1.71:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                                if psi > -1.71:
                                    next.agent_mode = AgentMode.SL
                                    next.timer = 0
                            if psi > -0.32:
                                next.agent_mode = AgentMode.SR
                                next.timer = 0
                        if rho > 2743.34:
                            if psi <= 0.57:
                                next.agent_mode = AgentMode.SR
                                next.timer = 0
                            if psi > 0.57:
                                if rho <= 4775.45:
                                    next.agent_mode = AgentMode.WR
                                    next.timer = 0
                                if rho > 4775.45:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
            if rho > 11074.98:
                if theta <= -1.27:
                    next.agent_mode = AgentMode.COC
                    next.timer = 0
                if theta > -1.27:
                    if theta <= 1.33:
                        if theta <= 0.51:
                            if psi <= 0.57:
                                if theta <= -0.19:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                                if theta > -0.19:
                                    if psi <= -1.21:
                                        if rho <= 15748.83:
                                            next.agent_mode = AgentMode.SR
                                            next.timer = 0
                                        if rho > 15748.83:
                                            next.agent_mode = AgentMode.WR
                                            next.timer = 0
                                    if psi > -1.21:
                                        if rho <= 35053.85:
                                            next.agent_mode = AgentMode.COC
                                            next.timer = 0
                                        if rho > 35053.85:
                                            next.agent_mode = AgentMode.SR
                                            next.timer = 0
                            if psi > 0.57:
                                if theta <= 0.13:
                                    if theta <= 0.0:
                                        if theta <= -0.7:
                                            if psi <= 1.9:
                                                next.agent_mode = AgentMode.WR
                                                next.timer = 0
                                            if psi > 1.9:
                                                next.agent_mode = AgentMode.COC
                                                next.timer = 0
                                        if theta > -0.7:
                                            if psi <= 1.4:
                                                next.agent_mode = AgentMode.COC
                                                next.timer = 0
                                            if psi > 1.4:
                                                next.agent_mode = AgentMode.WR
                                                next.timer = 0
                                    if theta > 0.0:
                                        next.agent_mode = AgentMode.WR
                                        next.timer = 0
                                if theta > 0.13:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                        if theta > 0.51:
                            if psi <= -0.25:
                                if psi <= -2.22:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                                if psi > -2.22:
                                    next.agent_mode = AgentMode.WR
                                    next.timer = 0
                            if psi > -0.25:
                                if psi <= 0.76:
                                    next.agent_mode = AgentMode.SR
                                    next.timer = 0
                                if psi > 0.76:
                                    next.agent_mode = AgentMode.COC
                                    next.timer = 0
                    if theta > 1.33:
                        next.agent_mode = AgentMode.COC
                        next.timer = 0

    return next
'''