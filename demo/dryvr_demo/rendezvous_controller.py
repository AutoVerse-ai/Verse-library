from enum import Enum, auto
import copy


class CraftMode(Enum):
    ProxA = auto()
    ProxB = auto()
    Passive = auto()


class State:
    xp = 0.0
    yp = 0.0
    xd = 0.0
    yd = 0.0
    total_time = 0.0
    cycle_time = 0.0
    craft_mode: CraftMode = CraftMode.ProxA

    def __init__(self, xp, yp, xd, yd, total_time, cycle_time, craft_mode: CraftMode):
        pass


def controller(ego: State):
    output = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.ProxA:
        if ego.yp >= -100 and ego.xp+ego.yp >= -141.1 and ego.xp >= -100 and ego.yp-ego.xp <= 141.1 and ego.yp <= 100 and ego.xp+ego.yp <= 141.1 and ego.xp <= 100 and ego.yp-ego.xp >= -141.1:
            output.craft_mode = CraftMode.ProxB
        if ego.cycle_time >= 120:
            output.craft_mode = CraftMode.Passive
            output.cycle_time = 0.0

    if ego.craft_mode == CraftMode.ProxB:
        if ego.cycle_time >= 120:
            output.craft_mode = CraftMode.Passive
            output.cycle_time = 0.0
    
    assert (ego.craft_mode!=CraftMode.ProxB or\
         (ego.xp>=-105 and ego.yp>=0.57735*ego.xp and -ego.yp>=0.57735*ego.xp)), "Line-of-sight"
    assert (ego.craft_mode!=CraftMode.Passive or\
         (ego.xp<=-0.2 or ego.xp>=0.2 or ego.yp<=-0.2 or ego.yp>=0.2)), "Collision avoidance"
    return output
