from enum import Enum, auto
import copy
from typing import List


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
    # theta2: float 
    psi: float
    # psi2: float
    v_own: float
    # v_int: float
    timer_DL: float
    agent_mode: AgentMode

    def __init__(self, rho, theta, psi, v_own, timer_DL, agent_mode: AgentMode):#__init__(self, x, y, theta, v, agent_mode: AgentMode):
        pass


def decisionLogic(ego: State, others: List[State]):
    next = copy.deepcopy(ego)
    rho = ego.rho
    theta = ego.theta
    psi = ego.psi
    v_int = ego.v_int
    v_own = ego.v_own
    
    
    pi = 3.14
    
    acas_update_time = 1
    
    if ego.timer_DL >= acas_update_time:
        if rho <= 5000.0:
            if rho <= 1666.67:
                if theta <= 0.0:
                    if v_own <= 350.0:
                        next.agent_mode = AgentMode.SL
                        next.timer_DL = 0
                    if v_own > 350.0:
                        next.agent_mode = AgentMode.SL
                        next.timer_DL = 0
                if theta > 0.0:
                    if psi <= 0.0:
                        if v_own <= 350.0:
                            next.agent_mode = AgentMode.SL
                            next.timer_DL = 0
                        if v_own > 350.0:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                    if psi > 0.0:
                        next.agent_mode = AgentMode.SL
                        next.timer_DL = 0
            if rho > 1666.67:
                if theta <= 0.0:
                    if psi <= 0.0:
                        if v_own <= 450.0:
                            next.agent_mode = AgentMode.WL
                            next.timer_DL = 0
                        if v_own > 450.0:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
                    if psi > 0.0:
                        if v_int <= 312.5:
                            if v_int <= 62.5:
                                next.agent_mode = AgentMode.WL
                                next.timer_DL = 0
                            if v_int > 62.5:
                                if v_own <= 350.0:
                                    next.agent_mode = AgentMode.WR
                                    next.timer_DL = 0
                                if v_own > 350.0:
                                    next.agent_mode = AgentMode.WL
                                    next.timer_DL = 0
                        if v_int > 312.5:
                            next.agent_mode = AgentMode.WL
                            next.timer_DL = 0
                if theta > 0.0:
                    if v_int <= 187.5:
                        next.agent_mode = AgentMode.WR
                        next.timer_DL = 0
                    if v_int > 187.5:
                        if psi <= 0.0:
                            if v_own <= 450.0:
                                next.agent_mode = AgentMode.WR
                                next.timer_DL = 0
                            if v_own > 450.0:
                                next.agent_mode = AgentMode.WL
                                next.timer_DL = 0
                        if psi > 0.0:
                            next.agent_mode = AgentMode.SR
                            next.timer_DL = 0
        if rho > 5000.0:
            next.agent_mode = AgentMode.COC
            next.timer_DL = 0
    assert rho > 0, "too close"
    return next