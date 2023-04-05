from enum import Enum, auto
import copy


class CraftMode(Enum):
    Approaching = auto()
    Rendezvous = auto()
    Aborting = auto()


class State:
    x = 0.0
    y = 0.0
    vx = 0.0
    vy = 0.0
    total_time = 0.0
    cycle_time = 0.0
    craft_mode: CraftMode = CraftMode.Approaching

    def __init__(self, x, y, vx, vy, total_time, cycle_time, craft_mode: CraftMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.Approaching:
        # if ego.yp >= -100 and ego.xp+ego.yp >= -141.1 and ego.xp >= -100 and ego.yp-ego.xp <= 141.1 and ego.yp <= 100 and ego.xp+ego.yp <= 141.1 and ego.xp <= 100 and ego.yp-ego.xp >= -141.1:
        #     output.craft_mode = CraftMode.ProxB
        # if ego.cycle_time >= 120:
        #     output.craft_mode = CraftMode.Passive
        #     output.cycle_time = 0.0
        if( ego.x >= -100):
            output.craft_mode = CraftMode.Rendezvous
            output.x = -100
        if(120<=ego.total_time and ego.total_time<=150):
            output.craft_mode = CraftMode.Aborting
    if ego.craft_mode == CraftMode.Rendezvous:
        if (120<= ego.total_time and ego.total_time<=150):
            output.craft_mode = CraftMode.Aborting
        # if ( -1000 <= ego.x < -100):
        #     output.craft_mode = CraftMode.Approaching
    # if ego.craft_mode == CraftMode.Aborting:
    #     if (ego.x >= -100):
    #         output.craft_mode = CraftMode.Rendezvous
    #     if ( -1000 <= ego.x < -100):
    #         output.craft_mode = CraftMode.Approaching
    
    assert (ego.craft_mode!=CraftMode.Rendezvous or\
         ego.x>=-100 and ego.y>=0.36397023426*ego.x and -1*ego.y>=0.36397023426*ego.x), "Line-of-sight"
    assert (ego.craft_mode != CraftMode.Rendezvous or \
            (ego.vx ** 2 + ego.vy ** 2) <= 10.89), "velocity constraint"
    assert (ego.craft_mode!=CraftMode.Aborting or\
         (ego.x <= -2 or ego.x>=2 or ego.y<=-2 or ego.y>=2)), "Collision avoidance"
    return output
