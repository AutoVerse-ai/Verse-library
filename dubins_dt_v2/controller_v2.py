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
    timer_DL: float
    agent_mode: AgentMode

    def __init__(self, rho, theta, psi, v_own, v_int, timer_DL, agent_mode: AgentMode):#__init__(self, x, y, theta, v, agent_mode: AgentMode):
        pass


def decisionLogic(ego: State, others: List[State]):
    next = copy.deepcopy(ego)
    rho = ego.rho
    theta = ego.theta
    psi = ego.psi
    
    pi = 3.14
    
    acas_update_time = 3.9
    
    # Variation that takes 6 min to run
    if ego.timer_DL >= acas_update_time:
        # Constrain angles to [-pi, pi]
        '''if theta < pi or theta > pi:
            scale_factor = (theta + pi) // (2 * pi)
            theta = theta - scale_factor * 2 * pi
            
        if psi < pi or psi > pi:
            scale_factor = (psi + pi) // (2 * pi)
            psi = psi - scale_factor * 2 * pi'''
        
        if ego.agent_mode == AgentMode.COC:  # advisory 0
            if rho <= 15139.2:
                if theta <= -0.22:
                    if rho <= 1727.29:
                        if psi <= 0.22:
                            if rho <= 1320.87:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if rho > 1320.87:
                                if theta <= -1.73:
                                    next.agent_mode = AgentMode.WL
                                    next.timer_DL = 0
                                if theta > -1.73:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                        if psi > 0.22:
                            if rho <= 711.24:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                            if rho > 711.24:
                                if theta <= -2.6:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                                if theta > -2.6:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                    if rho > 1727.29:
                        if theta <= -0.87:
                            if psi <= -0.22:
                                if rho <= 5385.08:
                                    next.agent_mode = AgentMode.WL
                                    next.timer_DL = 0
                                if rho > 5385.08:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                            if psi > -0.22:
                                if theta <= -2.82:
                                    if psi <= 0.43:
                                        next.agent_mode = AgentMode.WL
                                        next.timer_DL = 0
                                    if psi > 0.43:
                                        next.agent_mode = AgentMode.WR
                                        next.timer_DL = 0
                                if theta > -2.82:
                                    if rho <= 6197.93:
                                        if psi <= 1.95:
                                            if theta <= -2.17:
                                                next.agent_mode = AgentMode.WL
                                                next.timer_DL = 0
                                            if theta > -2.17:
                                                if rho <= 4369.03:
                                                    next.agent_mode = AgentMode.SL
                                                    next.timer_DL = 0
                                                if rho > 4369.03:
                                                    if psi <= 0.65:
                                                        next.agent_mode = AgentMode.WL
                                                        next.timer_DL = 0
                                                    if psi > 0.65:
                                                        next.agent_mode = AgentMode.SL
                                                        next.timer_DL = 0
                                        if psi > 1.95:
                                            next.agent_mode = AgentMode.WL
                                            next.timer_DL = 0
                                    if rho > 6197.93:
                                        next.agent_mode = AgentMode.WL
                                        next.timer_DL = 0
                        if theta > -0.87:
                            if rho <= 4369.03:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if rho > 4369.03:
                                if rho <= 9652.51:
                                    if psi <= 1.08:
                                        next.agent_mode = AgentMode.WL
                                        next.timer_DL = 0
                                    if psi > 1.08:
                                        if theta <= -0.43:
                                            next.agent_mode = AgentMode.SL
                                            next.timer_DL = 0
                                        if theta > -0.43:
                                            if psi <= 2.6:
                                                next.agent_mode = AgentMode.SR
                                                next.timer_DL = 0
                                            if psi > 2.6:
                                                next.agent_mode = AgentMode.SL
                                                next.timer_DL = 0
                                if rho > 9652.51:
                                    if psi <= 1.3:
                                        next.agent_mode = AgentMode.COC
                                        next.timer_DL = 0
                                    if psi > 1.3:
                                        next.agent_mode = AgentMode.WL
                                        next.timer_DL = 0
                if theta > -0.22:
                    if rho <= 1727.29:
                        if psi <= -0.22:
                            if rho <= 508.03:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if rho > 508.03:
                                if theta <= 0.43:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                                if theta > 0.43:
                                    if theta <= 2.82:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                                    if theta > 2.82:
                                        next.agent_mode = AgentMode.SL
                                        next.timer_DL = 0
                        if psi > -0.22:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                    if rho > 1727.29:
                        if theta <= 1.3:
                            if rho <= 5181.87:
                                if psi <= -0.22:
                                    if theta <= 0.43:
                                        next.agent_mode = AgentMode.SL
                                        next.timer_DL = 0
                                    if theta > 0.43:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                                if psi > -0.22:
                                    if theta <= 0.43:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                                    if theta > 0.43:
                                        if rho <= 3352.98:
                                            next.agent_mode = AgentMode.SR
                                            next.timer_DL = 0
                                        if rho > 3352.98:
                                            next.agent_mode = AgentMode.WR
                                            next.timer_DL = 0
                            if rho > 5181.87:
                                if psi <= -1.08:
                                    if theta <= 0.43:
                                        next.agent_mode = AgentMode.WL
                                        next.timer_DL = 0
                                    if theta > 0.43:
                                        if rho <= 8230.03:
                                            next.agent_mode = AgentMode.SR
                                            next.timer_DL = 0
                                        if rho > 8230.03:
                                            next.agent_mode = AgentMode.WR
                                            next.timer_DL = 0
                                if psi > -1.08:
                                    next.agent_mode = AgentMode.WR
                                    next.timer_DL = 0
                        if theta > 1.3:
                            if rho <= 8839.67:
                                if psi <= 0.43:
                                    if theta <= 2.82:
                                        next.agent_mode = AgentMode.WR
                                        next.timer_DL = 0
                                    if theta > 2.82:
                                        next.agent_mode = AgentMode.WR
                                        next.timer_DL = 0
                                if psi > 0.43:
                                    if rho <= 4369.03:
                                        next.agent_mode = AgentMode.WR
                                        next.timer_DL = 0
                                    if rho > 4369.03:
                                        next.agent_mode = AgentMode.COC
                                        next.timer_DL = 0
                            if rho > 8839.67:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
            if rho > 15139.2:
                if rho <= 18593.78:
                    if psi <= 1.95:
                        next.agent_mode = AgentMode.COC
                        next.timer_DL = 0
                    if psi > 1.95:
                        next.agent_mode = AgentMode.WR
                        next.timer_DL = 0
                if rho > 18593.78:
                    next.agent_mode = AgentMode.COC
                    next.timer_DL = 0
            
        if ego.agent_mode == AgentMode.WL:
            if psi <= -3.03:
                if theta <= 0.22:
                    if rho <= 13107.09:
                        if theta <= 0.0:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if theta > 0.0:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                    if rho > 13107.09:
                        if theta <= -0.43:
                            next.agent_mode = AgentMode.COC
                            next.timer_DL = 0
                        if theta > -0.43:
                            next.agent_mode = AgentMode.WL
                            next.timer_DL = 0
                if theta > 0.22:
                    if rho <= 10465.35:
                        next.agent_mode = AgentMode.SR
                        next.timer_DL = 0
                    if rho > 10465.35:
                        if rho <= 10668.56:
                            next.agent_mode = AgentMode.WR
                            next.timer_DL = 0
                        if rho > 10668.56:
                            next.agent_mode = AgentMode.COC
                            next.timer_DL = 0
            if psi > -3.03:
                if rho <= 9246.09:
                    if theta <= -0.43:
                        if rho <= 1930.5:
                            if psi <= 0.22:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if psi > 0.22:
                                if psi <= 1.52:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                                if psi > 1.52:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                        if rho > 1930.5:
                            if psi <= -0.43:
                                next.agent_mode = AgentMode.WL
                                next.timer_DL = 0
                            if psi > -0.43:
                                if theta <= -1.3:
                                    if rho <= 3759.4:
                                        if psi <= 1.73:
                                            next.agent_mode = AgentMode.SL
                                            next.timer_DL = 0
                                        if psi > 1.73:
                                            next.agent_mode = AgentMode.WL
                                            next.timer_DL = 0
                                    if rho > 3759.4:
                                        next.agent_mode = AgentMode.WL
                                        next.timer_DL = 0
                                if theta > -1.3:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                    if theta > -0.43:
                        if theta <= 1.95:
                            if psi <= 0.0:
                                if theta <= 0.43:
                                    if psi <= -2.6:
                                        if theta <= 0.22:
                                            next.agent_mode = AgentMode.SL
                                            next.timer_DL = 0
                                        if theta > 0.22:
                                            next.agent_mode = AgentMode.SR
                                            next.timer_DL = 0
                                    if psi > -2.6:
                                        next.agent_mode = AgentMode.SL
                                        next.timer_DL = 0
                                if theta > 0.43:
                                    if rho <= 1320.87:
                                        next.agent_mode = AgentMode.SL
                                        next.timer_DL = 0
                                    if rho > 1320.87:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                            if psi > 0.0:
                                if theta <= 1.08:
                                    if theta <= -0.22:
                                        if psi <= 2.38:
                                            next.agent_mode = AgentMode.SR
                                            next.timer_DL = 0
                                        if psi > 2.38:
                                            next.agent_mode = AgentMode.SL
                                            next.timer_DL = 0
                                    if theta > -0.22:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                                if theta > 1.08:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                        if theta > 1.95:
                            if rho <= 1727.29:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if rho > 1727.29:
                                if rho <= 3962.61:
                                    next.agent_mode = AgentMode.WL
                                    next.timer_DL = 0
                                if rho > 3962.61:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                if rho > 9246.09:
                    if theta <= 0.65:
                        if theta <= -1.08:
                            if rho <= 13716.72:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
                            if rho > 13716.72:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
                        if theta > -1.08:
                            if psi <= 0.65:
                                if psi <= -1.52:
                                    if theta <= -0.22:
                                        next.agent_mode = AgentMode.COC
                                        next.timer_DL = 0
                                    if theta > -0.22:
                                        if rho <= 15342.41:
                                            next.agent_mode = AgentMode.WL
                                            next.timer_DL = 0
                                        if rho > 15342.41:
                                            next.agent_mode = AgentMode.WL
                                            next.timer_DL = 0
                                if psi > -1.52:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                            if psi > 0.65:
                                if theta <= 0.0:
                                    next.agent_mode = AgentMode.WL
                                    next.timer_DL = 0
                                if theta > 0.0:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                    if theta > 0.65:
                        next.agent_mode = AgentMode.COC
                        next.timer_DL = 0
                        
        if ego.agent_mode == AgentMode.WR:
            if rho <= 7417.19:
                if theta <= 0.43:
                    if theta <= -2.17:
                        if rho <= 1524.08:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                        if rho > 1524.08:
                            next.agent_mode = AgentMode.WR
                            next.timer_DL = 0
                    if theta > -2.17:
                        if theta <= -0.65:
                            if rho <= 914.45:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                            if rho > 914.45:
                                if psi <= -0.22:
                                    next.agent_mode = AgentMode.WR
                                    next.timer_DL = 0
                                if psi > -0.22:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                        if theta > -0.65:
                            if psi <= -0.22:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if psi > -0.22:
                                if theta <= -0.22:
                                    if psi <= 2.17:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                                    if psi > 2.17:
                                        next.agent_mode = AgentMode.SL
                                        next.timer_DL = 0
                                if theta > -0.22:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                if theta > 0.43:
                    if rho <= 1727.29:
                        if psi <= -0.22:
                            if psi <= -1.95:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                            if psi > -1.95:
                                if theta <= 0.87:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                                if theta > 0.87:
                                    if theta <= 2.38:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                                    if theta > 2.38:
                                        next.agent_mode = AgentMode.SL
                                        next.timer_DL = 0
                        if psi > -0.22:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                    if rho > 1727.29:
                        if theta <= 1.52:
                            if psi <= 0.0:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                            if psi > 0.0:
                                next.agent_mode = AgentMode.WR
                                next.timer_DL = 0
                        if theta > 1.52:
                            next.agent_mode = AgentMode.WR
                            next.timer_DL = 0
            if rho > 7417.19:
                if rho <= 7823.61:
                    if theta <= -0.65:
                        next.agent_mode = AgentMode.WL
                        next.timer_DL = 0
                    if theta > -0.65:
                        next.agent_mode = AgentMode.WR
                        next.timer_DL = 0
                if rho > 7823.61:
                    if theta <= -1.08:
                        next.agent_mode = AgentMode.COC
                        next.timer_DL = 0
                    if theta > -1.08:
                        if theta <= 1.08:
                            if rho <= 11481.4:
                                next.agent_mode = AgentMode.WR
                                next.timer_DL = 0
                            if rho > 11481.4:
                                if theta <= 0.65:
                                    if psi <= 0.65:
                                        if psi <= -1.73:
                                            if theta <= -0.22:
                                                next.agent_mode = AgentMode.COC
                                                next.timer_DL = 0
                                            if theta > -0.22:
                                                next.agent_mode = AgentMode.WR
                                                next.timer_DL = 0
                                        if psi > -1.73:
                                            next.agent_mode = AgentMode.COC
                                            next.timer_DL = 0
                                    if psi > 0.65:
                                        if theta <= 0.22:
                                            if theta <= -0.65:
                                                if psi <= 1.95:
                                                    next.agent_mode = AgentMode.WR
                                                    next.timer_DL = 0
                                                if psi > 1.95:
                                                    next.agent_mode = AgentMode.COC
                                                    next.timer_DL = 0
                                            if theta > -0.65:
                                                next.agent_mode = AgentMode.WR
                                                next.timer_DL = 0
                                        if theta > 0.22:
                                            next.agent_mode = AgentMode.COC
                                            next.timer_DL = 0
                                if theta > 0.65:
                                    if psi <= -0.87:
                                        if rho <= 27128.63:
                                            next.agent_mode = AgentMode.WR
                                            next.timer_DL = 0
                                        if rho > 27128.63:
                                            next.agent_mode = AgentMode.COC
                                            next.timer_DL = 0
                                    if psi > -0.87:
                                        next.agent_mode = AgentMode.COC
                                        next.timer_DL = 0
                        if theta > 1.08:
                            if rho <= 11887.83:
                                if psi <= 0.0:
                                    next.agent_mode = AgentMode.WR
                                    next.timer_DL = 0
                                if psi > 0.0:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                            if rho > 11887.83:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
                        
        if ego.agent_mode == AgentMode.SL:
            if rho <= 9042.88:
                if rho <= 5385.08:
                    if theta <= -0.43:
                        if psi <= 0.0:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if psi > 0.0:
                            if rho <= 914.45:
                                if psi <= 1.52:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                                if psi > 1.52:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                            if rho > 914.45:
                                if theta <= -0.65:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                                if theta > -0.65:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                    if theta > -0.43:
                        if rho <= 711.24:
                            if psi <= -0.22:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if psi > -0.22:
                                if theta <= 0.87:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                                if theta > 0.87:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                        if rho > 711.24:
                            if theta <= 0.0:
                                if psi <= 0.0:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                                if psi > 0.0:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                            if theta > 0.0:
                                if theta <= 2.38:
                                    if theta <= 0.43:
                                        if psi <= -0.43:
                                            if psi <= -2.6:
                                                next.agent_mode = AgentMode.SR
                                                next.timer_DL = 0
                                            if psi > -2.6:
                                                next.agent_mode = AgentMode.SL
                                                next.timer_DL = 0
                                        if psi > -0.43:
                                            next.agent_mode = AgentMode.SR
                                            next.timer_DL = 0
                                    if theta > 0.43:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                                if theta > 2.38:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                if rho > 5385.08:
                    if rho <= 5588.29:
                        if theta <= 0.87:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if theta > 0.87:
                            if psi <= -0.65:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                            if psi > -0.65:
                                next.agent_mode = AgentMode.WR
                                next.timer_DL = 0
                    if rho > 5588.29:
                        if theta <= -0.22:
                            if theta <= -1.73:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
                            if theta > -1.73:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                        if theta > -0.22:
                            if theta <= 1.52:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                            if theta > 1.52:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
            if rho > 9042.88:
                if theta <= -1.3:
                    next.agent_mode = AgentMode.COC
                    next.timer_DL = 0
                if theta > -1.3:
                    if theta <= 1.08:
                        if rho <= 14123.14:
                            if psi <= -1.73:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if psi > -1.73:
                                if psi <= 2.17:
                                    next.agent_mode = AgentMode.WL
                                    next.timer_DL = 0
                                if psi > 2.17:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                        if rho > 14123.14:
                            if rho <= 36476.32:
                                if psi <= 0.43:
                                    if theta <= -0.22:
                                        if psi <= -1.3:
                                            next.agent_mode = AgentMode.COC
                                            next.timer_DL = 0
                                        if psi > -1.3:
                                            next.agent_mode = AgentMode.SL
                                            next.timer_DL = 0
                                    if theta > -0.22:
                                        if psi <= -0.22:
                                            next.agent_mode = AgentMode.WL
                                            next.timer_DL = 0
                                        if psi > -0.22:
                                            next.agent_mode = AgentMode.COC
                                            next.timer_DL = 0
                                if psi > 0.43:
                                    if theta <= 0.22:
                                        next.agent_mode = AgentMode.WL
                                        next.timer_DL = 0
                                    if theta > 0.22:
                                        next.agent_mode = AgentMode.COC
                                        next.timer_DL = 0
                            if rho > 36476.32:
                                if theta <= -1.08:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                                if theta > -1.08:
                                    if psi <= 1.08:
                                        if psi <= -0.87:
                                            if theta <= -0.22:
                                                next.agent_mode = AgentMode.COC
                                                next.timer_DL = 0
                                            if theta > -0.22:
                                                next.agent_mode = AgentMode.WL
                                                next.timer_DL = 0
                                        if psi > -0.87:
                                            if theta <= -0.87:
                                                next.agent_mode = AgentMode.WL
                                                next.timer_DL = 0
                                            if theta > -0.87:
                                                next.agent_mode = AgentMode.COC
                                                next.timer_DL = 0
                                    if psi > 1.08:
                                        if theta <= 0.22:
                                            next.agent_mode = AgentMode.WL
                                            next.timer_DL = 0
                                        if theta > 0.22:
                                            next.agent_mode = AgentMode.COC
                                            next.timer_DL = 0
                    if theta > 1.08:
                        if theta <= 1.52:
                            if psi <= -0.22:
                                if psi <= -1.08:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                                if psi > -1.08:
                                    next.agent_mode = AgentMode.WL
                                    next.timer_DL = 0
                            if psi > -0.22:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
                        if theta > 1.52:
                            next.agent_mode = AgentMode.COC
                            next.timer_DL = 0
                    
        if ego.agent_mode == AgentMode.SR:
            if rho <= 10668.56:
                if theta <= -1.08:
                    if rho <= 5588.29:
                        if rho <= 1524.08:
                            if psi <= 0.43:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if psi > 0.43:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                        if rho > 1524.08:
                            if psi <= -0.22:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
                            if psi > -0.22:
                                if psi <= 1.95:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                                if psi > 1.95:
                                    next.agent_mode = AgentMode.WR
                                    next.timer_DL = 0
                    if rho > 5588.29:
                        if theta <= -1.73:
                            next.agent_mode = AgentMode.COC
                            next.timer_DL = 0
                        if theta > -1.73:
                            if psi <= 1.08:
                                if psi <= 0.22:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                                if psi > 0.22:
                                    if psi <= 0.65:
                                        next.agent_mode = AgentMode.WL
                                        next.timer_DL = 0
                                    if psi > 0.65:
                                        if rho <= 7823.61:
                                            next.agent_mode = AgentMode.SL
                                            next.timer_DL = 0
                                        if rho > 7823.61:
                                            next.agent_mode = AgentMode.WL
                                            next.timer_DL = 0
                            if psi > 1.08:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                if theta > -1.08:
                    if theta <= 0.43:
                        if psi <= -0.22:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if psi > -0.22:
                            if theta <= -0.22:
                                if rho <= 1524.08:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                                if rho > 1524.08:
                                    if theta <= -0.65:
                                        next.agent_mode = AgentMode.SL
                                        next.timer_DL = 0
                                    if theta > -0.65:
                                        if psi <= 2.17:
                                            next.agent_mode = AgentMode.SR
                                            next.timer_DL = 0
                                        if psi > 2.17:
                                            next.agent_mode = AgentMode.SL
                                            next.timer_DL = 0
                            if theta > -0.22:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                    if theta > 0.43:
                        if rho <= 2743.34:
                            if psi <= -0.22:
                                if theta <= 0.87:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                                if theta > 0.87:
                                    if psi <= -1.08:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                                    if psi > -1.08:
                                        next.agent_mode = AgentMode.SL
                                        next.timer_DL = 0
                            if psi > -0.22:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                        if rho > 2743.34:
                            if psi <= 0.65:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                            if psi > 0.65:
                                if theta <= 1.52:
                                    next.agent_mode = AgentMode.WR
                                    next.timer_DL = 0
                                if theta > 1.52:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
            if rho > 10668.56:
                if theta <= -1.3:
                    next.agent_mode = AgentMode.COC
                    next.timer_DL = 0
                if theta > -1.3:
                    if theta <= 1.3:
                        if psi <= -0.87:
                            if theta <= -0.22:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
                            if theta > -0.22:
                                if theta <= 1.08:
                                    next.agent_mode = AgentMode.WR
                                    next.timer_DL = 0
                                if theta > 1.08:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                        if psi > -0.87:
                            if theta <= 0.0:
                                if psi <= 0.43:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                                if psi > 0.43:
                                    if theta <= -0.65:
                                        if psi <= 1.95:
                                            next.agent_mode = AgentMode.WR
                                            next.timer_DL = 0
                                        if psi > 1.95:
                                            next.agent_mode = AgentMode.COC
                                            next.timer_DL = 0
                                    if theta > -0.65:
                                        if psi <= 1.52:
                                            next.agent_mode = AgentMode.COC
                                            next.timer_DL = 0
                                        if psi > 1.52:
                                            next.agent_mode = AgentMode.WR
                                            next.timer_DL = 0
                            if theta > 0.0:
                                if psi <= 0.65:
                                    if psi <= -0.22:
                                        if theta <= 0.87:
                                            next.agent_mode = AgentMode.SR
                                            next.timer_DL = 0
                                        if theta > 0.87:
                                            next.agent_mode = AgentMode.WR
                                            next.timer_DL = 0
                                    if psi > -0.22:
                                        next.agent_mode = AgentMode.SR
                                        next.timer_DL = 0
                                if psi > 0.65:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                    if theta > 1.3:
                        if rho <= 25909.37:
                            if psi <= 0.22:
                                if psi <= -0.65:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                                if psi > -0.65:
                                    next.agent_mode = AgentMode.WR
                                    next.timer_DL = 0
                            if psi > 0.22:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
                        if rho > 25909.37:
                            next.agent_mode = AgentMode.COC
                            next.timer_DL = 0
            
            
    return next
'''
def decisionLogic(ego: State, others: List[State]):
    output = copy.deepcopy(ego)
    # assert not vehicle_close(ego, others)
    #if ego.timer >= 2.1:
    #    output.timer = 0
    if ego.agent_mode!=AgentMode.SL and ego.timer >= 8.1:
        
        output.timer = 0
        #Logic 1: just using rho (and timer to try and avoid branching)
        if ego.timer >= 8.1 and ego.rho <= 15139.2 and ego.theta <= -0.22:
            output.agent_mode = AgentMode.SL
        if ego.timer >= 8.1 and ego.rho <= 15139.2 and ego.theta > -0.22:
            output.agent_mode = AgentMode.SR 
            #Logic 2:  Uncomment one more if-statement (for theta) and it does a lot of branching
            #if ego.timer >= 8.1 and ego.theta <= -0.22: # For logic 2
            #    output.agent_mode = AgentMode.SR # for logic 2
            
            #    if ego.timer >= 8.1 and ego.rho <= 1727.29:
            #        if ego.timer >= 8.1 and ego.psi <= 0.22:
            #            output.agent_mode = AgentMode.SL
    #if ego.rho >= 1000:
    #    output.agent_mode = AgentMode.WR
    return output


# This hasa tree version of the ACAS Xu collision avoidance logic
def decisionLogic(ego: State, others: List[State]):
    next = copy.deepcopy(ego)
    rho = ego.rho
    theta = ego.theta
    psi = ego.psi
    
    acas_update_time = 3.9
    
    # Variation that takes 6 min to run
    if ego.timer_DL >= acas_update_time:
        # Update timer
        
        if ego.agent_mode == AgentMode.COC:  # advisory 0
            if rho <= 15139.2:
                if theta <= -0.22:
                    if rho <= 1727.29:
                        if psi <= 0.22:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if psi > 0.22:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                    if rho > 1727.29:
                        next.agent_mode = AgentMode.WL
                        next.timer_DL = 0
                if theta > -0.22:
                    if rho <= 1727.29:
                        if psi <= -0.22:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if psi > -0.22:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                    if rho > 1727.29:
                        if theta <= 1.3:
                            if rho <= 5181.87:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                            if rho > 5181.87:
                                next.agent_mode = AgentMode.WR
                                next.timer_DL = 0
                        if theta > 1.3:
                            next.agent_mode = AgentMode.WR
                            next.timer_DL = 0
            if rho > 15139.2:
                next.agent_mode = AgentMode.COC
                next.timer_DL = 0
                   
        if ego.agent_mode == AgentMode.WL: # advisory 1
            if psi <= -3.03:
                if theta <= 0.22:
                    next.agent_mode = AgentMode.WL
                    next.timer_DL = 0
                if theta > 0.22:
                    if rho <= 10465.35:
                        next.agent_mode = AgentMode.SR
                        next.timer_DL = 0
                    if rho > 10465.35:
                        next.agent_mode = AgentMode.WR
                        next.timer_DL = 0
            if psi > -3.03:
                if rho <= 9246.09:
                    if theta <= -0.43:
                        if rho <= 1930.5:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if rho > 1930.5:
                            next.agent_mode = AgentMode.WL
                            next.timer_DL = 0
                    if theta > -0.43:
                        if theta <= 1.95:
                            if psi <= 0.0:
                                if theta <= 0.43:
                                    next.agent_mode = AgentMode.SL
                                    next.timer_DL = 0
                                if theta > 0.43:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                            if psi > 0.0:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                        if theta > 1.95:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                if rho > 9246.09:
                    if theta <= 0.65:
                        if theta <= -1.08:
                            next.agent_mode = AgentMode.COC
                            next.timer_DL = 0
                        if theta > -1.08:
                            next.agent_mode = AgentMode.WL
                            next.timer_DL = 0
                    if theta > 0.65:
                        next.agent_mode = AgentMode.COC
                        next.timer_DL = 0
        
        if ego.agent_mode == AgentMode.WR: # advisory 2
            if rho <= 7417.19:
                if theta <= 0.43:
                    if theta <= -2.17:
                        next.agent_mode = AgentMode.SR
                        next.timer_DL = 0
                    if theta > -2.17:
                        if theta <= -0.65:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if theta > -0.65:
                            if psi <= -0.22:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if psi > -0.22:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                if theta > 0.43:
                    if rho <= 1727.29:
                        next.agent_mode = AgentMode.SR
                        next.timer_DL = 0
                    if rho > 1727.29:
                        next.agent_mode = AgentMode.WR
                        next.timer_DL = 0
            if rho > 7417.19:
                if rho <= 7823.61:
                    next.agent_mode = AgentMode.WL
                    next.timer_DL = 0
                if rho > 7823.61:
                    if theta <= -1.08:
                        next.agent_mode = AgentMode.COC
                        next.timer_DL = 0
                    if theta > -1.08:
                        if theta <= 1.08:
                            next.agent_mode = AgentMode.WR
                            next.timer_DL = 0
                        if theta > 1.08:
                            next.agent_mode = AgentMode.COC
                            next.timer_DL = 0

        if ego.agent_mode == AgentMode.SL: # advisory 3
            if rho <= 9042.88:
                if rho <= 5385.08:
                    if theta <= -0.43:
                        next.agent_mode = AgentMode.SL
                        next.timer_DL = 0
                    if theta > -0.43:
                        if rho <= 711.24:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if rho > 711.24:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                if rho > 5385.08:
                    if rho <= 5588.29:
                        next.agent_mode = AgentMode.WR
                        next.timer_DL = 0
                    if rho > 5588.29:
                        next.agent_mode = AgentMode.SR
                        next.timer_DL = 0
            if rho > 9042.88:
                if theta <= -1.3:
                    next.agent_mode = AgentMode.COC
                    next.timer_DL = 0
                if theta > -1.3:
                    if theta <= 1.08:
                        next.agent_mode = AgentMode.WL
                        next.timer_DL = 0
                    if theta > 1.08:
                        next.agent_mode = AgentMode.COC
                        next.timer_DL = 0     
        
        if ego.agent_mode == AgentMode.SR: # advisory 4
            if rho <= 10668.56:
                if theta <= -1.08:
                    if rho <= 5588.29:
                        if rho <= 1524.08:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                        if rho > 1524.08:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                    if rho > 5588.29:
                        next.agent_mode = AgentMode.WL
                        next.timer_DL = 0
                if theta > -1.08:
                    if theta <= 0.43:
                        if psi <= -0.22:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if psi > -0.22:
                            if theta <= -0.22:
                                next.agent_mode = AgentMode.SL
                                next.timer_DL = 0
                            if theta > -0.22:
                                next.agent_mode = AgentMode.SR
                                next.timer_DL = 0
                    if theta > 0.43:
                        next.agent_mode = AgentMode.SR
                        next.timer_DL = 0
            if rho > 10668.56:
                if theta <= -1.3:
                    next.agent_mode = AgentMode.COC
                    next.timer_DL = 0
                if theta > -1.3:
                    if theta <= 1.3:
                        if psi <= -0.87:
                            if theta <= -0.22:
                                next.agent_mode = AgentMode.COC
                                next.timer_DL = 0
                            if theta > -0.22:
                                next.agent_mode = AgentMode.WR
                                next.timer_DL = 0
                        if psi > -0.87:
                            if theta <= 0.0:
                                next.agent_mode = AgentMode.WR
                                next.timer_DL = 0
                            if theta > 0.0:
                                if psi <= 0.65:
                                    next.agent_mode = AgentMode.SR
                                    next.timer_DL = 0
                                if psi > 0.65:
                                    next.agent_mode = AgentMode.COC
                                    next.timer_DL = 0
                    if theta > 1.3:
                        next.agent_mode = AgentMode.COC
                        next.timer_DL = 0

                        
    return next
'''