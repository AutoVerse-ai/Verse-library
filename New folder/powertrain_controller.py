from enum import Enum, auto
import copy


class CraftMode(Enum):
    negAngle = auto()
    deadzone = auto()
    posAngle = auto()
    negAngleInit = auto()


class State:
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    x4 = 0.0
    x5 = 0.0
    x6 =0.0
    x7 =0.0
    x8 =0.0
    x9 =0.0

    total_time = 0.0
    craft_mode: CraftMode = CraftMode.negAngleInit

    def __init__(self, x1, x2, x3, x4, x5,x6,x7,x8,x9, total_time, craft_mode: CraftMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.negAngleInit:
        # if ego.yp >= -100 and ego.xp+ego.yp >= -141.1 and ego.xp >= -100 and ego.yp-ego.xp <= 141.1 and ego.yp <= 100 and ego.xp+ego.yp <= 141.1 and ego.xp <= 100 and ego.yp-ego.xp >= -141.1:
        #     output.craft_mode = CraftMode.ProxB
        # if ego.cycle_time >= 120:
        #     output.craft_mode = CraftMode.Passive
        #     output.cycle_time = 0.0
        if( ego.total_time >.2):
            output.craft_mode = CraftMode.negAngle
    if ego.craft_mode == CraftMode.negAngle:

        if(ego.x1 > -.03):
            output.craft_mode = CraftMode.deadzone
    if ego.craft_mode == CraftMode.deadzone:
        if (ego.x1 >= .03):
            output.craft_mode = CraftMode.posAngle
    if ego.craft_mode == CraftMode.deadzone:
        if (ego.x1 < -.03):
            output.craft_mode = CraftMode.negAngle

    
    # assert (ego.craft_mode!=CraftMode.Rendezvous or\
    #      ego.x>=-100 and ego.y>=0.36397023426*ego.x and -ego.y>=0.36397023426*ego.x), "Line-of-sight"
    # assert (ego.craft_mode != CraftMode.Rendezvous or \
    #         (ego.vx ** 2 + ego.vy ** 2) ** .5 <= 3.3), "velocity constraint"
    # assert (ego.craft_mode!=CraftMode.Aborting or\
    #      (ego.x<=-2 or ego.x>=2 or ego.y<=-2 or ego.y>=2)), "Collision avoidance"
    return output
