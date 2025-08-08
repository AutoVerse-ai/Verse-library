from enum import Enum, auto
import copy
from typing import List
import numpy as np
import math


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

    t = math.sqrt 
    
    acas_update_time = 1
    
    if ego.timer_DL >= acas_update_time:

        if ego.agent_mode == AgentMode.COC:
            if rho <= 19904.14:
                if rho <= 1047.59:
                    if psi <= 0.0:
                        next.agent_mode = AgentMode.SL
                        # next.timer_DL = 0
                    if psi > 0.0:
                        next.agent_mode = AgentMode.SR
                        # next.timer_DL = 0
                if rho > 1047.59:
                    if theta <= 0.0:
                        next.agent_mode = AgentMode.WL
                        # next.timer_DL = 0
                    if theta > 0.0:
                        next.agent_mode = AgentMode.WR
                        # next.timer_DL = 0
            # if rho > 19904.14:
            #     if theta <= 1.4:
            #         if theta <= -0.7:
            #             next.agent_mode = AgentMode.COC
            #             # next.timer_DL = 0
            #         if theta > -0.7:
            #             next.agent_mode = AgentMode.COC
            #             # next.timer_DL = 0
            #     if theta > 1.4:
            #         next.agent_mode = AgentMode.COC
            #         # next.timer_DL = 0


        if ego.agent_mode == AgentMode.WL:
            if rho <= 13618.62:
                if rho <= 7333.1:
                    if rho <= 1047.59:
                        next.agent_mode = AgentMode.SL
                        # next.timer_DL = 0
                    if rho > 1047.59:
                        # if theta <= -1.4:
                        #     next.agent_mode = AgentMode.WL
                        #     # next.timer_DL = 0
                        if theta > -1.4:
                            next.agent_mode = AgentMode.SR
                            # next.timer_DL = 0
                if rho > 7333.1:
                    if v_int <= 200.0:
                        next.agent_mode = AgentMode.WR
                        # next.timer_DL = 0
                    if v_int > 200.0:
                        next.agent_mode = AgentMode.SR
                        # next.timer_DL = 0
            if rho > 13618.62:
                if theta <= 1.4:
                    if theta <= -1.4:
                        next.agent_mode = AgentMode.COC
                        # next.timer_DL = 0
                    # if theta > -1.4:
                    #     next.agent_mode = AgentMode.WL
                    #     # next.timer_DL = 0
                if theta > 1.4:
                    next.agent_mode = AgentMode.COC
                    # next.timer_DL = 0

        if ego.agent_mode == AgentMode.WR:
            if v_int <= 200.0:
                if v_own <= 772.22:
                    if rho <= 11523.45:
                        next.agent_mode = AgentMode.SL
                        # next.timer_DL = 0
                    if rho > 11523.45:
                        next.agent_mode = AgentMode.COC
                        # next.timer_DL = 0
                if v_own > 772.22:
                    if v_own <= 894.44:
                        next.agent_mode = AgentMode.WL
                        # next.timer_DL = 0
                    if v_own > 894.44:
                        next.agent_mode = AgentMode.SL
                        # next.timer_DL = 0
            if v_int > 200.0:
                if rho <= 17808.96:
                    if rho <= 1047.59:
                        next.agent_mode = AgentMode.SR
                        # next.timer_DL = 0
                    if rho > 1047.59:
                        # if rho <= 3142.76:
                        #     next.agent_mode = AgentMode.WR
                        #     # next.timer_DL = 0
                        if rho > 3142.76:
                            next.agent_mode = AgentMode.SL
                            # next.timer_DL = 0
                if rho > 17808.96:
                    next.agent_mode = AgentMode.COC
                    # next.timer_DL = 0

        if ego.agent_mode == AgentMode.SL:
            if v_int <= 66.67:
                if rho <= 7333.1:
                    next.agent_mode = AgentMode.SR
                    # next.timer_DL = 0
                if rho > 7333.1:
                    if rho <= 11523.45:
                        next.agent_mode = AgentMode.WR
                        # next.timer_DL = 0
                    if rho > 11523.45:
                        next.agent_mode = AgentMode.COC
                        # next.timer_DL = 0
            if v_int > 66.67:
                if rho <= 17808.96:
                    # if rho <= 3142.76:
                    #     next.agent_mode = AgentMode.SL
                    #     # next.timer_DL = 0
                    if rho > 3142.76:
                        if theta <= 0.7:
                            next.agent_mode = AgentMode.SR
                            # next.timer_DL = 0
                        if theta > 0.7:
                            next.agent_mode = AgentMode.WR
                            # next.timer_DL = 0
                if rho > 17808.96:
                    next.agent_mode = AgentMode.COC
                    # next.timer_DL = 0

        
        if ego.agent_mode == AgentMode.SR:
            if rho <= 11523.45:
                if rho <= 7333.1:
                    # if rho <= 1047.59:
                    #     next.agent_mode = AgentMode.SR
                    #     # next.timer_DL = 0
                    if rho > 1047.59:
                        next.agent_mode = AgentMode.SL
                        # next.timer_DL = 0
                if rho > 7333.1:
                    if theta <= -0.7:
                        if theta <= -1.4:
                            next.agent_mode = AgentMode.COC
                            # next.timer_DL = 0
                        if theta > -1.4:
                            if psi <= 0.7:
                                next.agent_mode = AgentMode.WL
                                # next.timer_DL = 0
                            if psi > 0.7:
                                next.agent_mode = AgentMode.SL
                                # next.timer_DL = 0
                    if theta > -0.7:
                        next.agent_mode = AgentMode.SL
                        # next.timer_DL = 0
            if rho > 11523.45:
                if theta <= 1.4:
                    if theta <= -1.4:
                        next.agent_mode = AgentMode.COC
                        # next.timer_DL = 0
                    if theta > -1.4:
                        next.agent_mode = AgentMode.WR
                        # next.timer_DL = 0
                if theta > 1.4:
                    next.agent_mode = AgentMode.COC
                    # next.timer_DL = 0


        
    # assert (ego.agent_mode == next.agent_mode) or (rho * np.sin(theta) >= 500 and rho * np.cos(theta) >= 100 )
    assert rho > 500, "too close"
    return next



# For 484
'''
if rho <= 5000.0:
        if rho <= 1666.67:
            if theta <= 0.0:
                if v_own <= 350.0 and ego.agent_mode != AgentMode.SL:
                    next.agent_mode = AgentMode.SL
                    # next.timer_DL = 0
                if v_own > 350.0 and ego.agent_mode != AgentMode.SL:
                    next.agent_mode = AgentMode.SL
                    # next.timer_DL = 0
            if theta > 0.0:
                if psi <= 0.0:
                    if v_own <= 350.0 and ego.agent_mode != AgentMode.SL:
                        next.agent_mode = AgentMode.SL
                        # next.timer_DL = 0
                    if v_own > 350.0 and ego.agent_mode != AgentMode.SR:
                        next.agent_mode = AgentMode.SR
                        # next.timer_DL = 0
                if psi > 0.0 and ego.agent_mode != AgentMode.SL:
                    next.agent_mode = AgentMode.SL
                    # next.timer_DL = 0
        if rho > 1666.67:
            if theta <= 0.0:
                if psi <= 0.0:
                    if v_own <= 450.0 and ego.agent_mode != AgentMode.WL:
                        next.agent_mode = AgentMode.WL
                        # next.timer_DL = 0
                    if v_own > 450.0 and ego.agent_mode != AgentMode.SR:
                        next.agent_mode = AgentMode.SR
                        # next.timer_DL = 0
                if psi > 0.0:
                    if v_int <= 312.5:
                        if v_int <= 62.5 and ego.agent_mode != AgentMode.WL:
                            next.agent_mode = AgentMode.WL
                            # next.timer_DL = 0
                        if v_int > 62.5:
                            if v_own <= 350.0 and ego.agent_mode != AgentMode.WR:
                                next.agent_mode = AgentMode.WR
                                # next.timer_DL = 0
                            if v_own > 350.0 and ego.agent_mode != AgentMode.WL:
                                next.agent_mode = AgentMode.WL
                                # next.timer_DL = 0
                    if v_int > 312.5 and ego.agent_mode != AgentMode.WL:
                        next.agent_mode = AgentMode.WL
                        # next.timer_DL = 0
            if theta > 0.0:
                if v_int <= 187.5 and ego.agent_mode != AgentMode.WR:
                    next.agent_mode = AgentMode.WR
                    # next.timer_DL = 0
                if v_int > 187.5:
                    if psi <= 0.0:
                        if v_own <= 450.0 and ego.agent_mode != AgentMode.WR:
                            next.agent_mode = AgentMode.WR
                            # next.timer_DL = 0
                        if v_own > 450.0 and ego.agent_mode != AgentMode.WL:
                            next.agent_mode = AgentMode.WL
                            # next.timer_DL = 0
                    if psi > 0.0 and ego.agent_mode != AgentMode.SR:
                        next.agent_mode = AgentMode.SR
                        # next.timer_DL = 0
    if rho > 5000.0 and ego.agent_mode != AgentMode.COC:
        next.agent_mode = AgentMode.COC
        # next.timer_DL = 0

'''