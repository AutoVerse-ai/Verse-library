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
    t = 0.0
    cycle_time = 0.0
    craft_mode: CraftMode = CraftMode.Approaching

    def __init__(self, x, y, vx, vy, ux,uy, t, cycle_time, craft_mode: CraftMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.Approaching:
        if( ego.y > -100 and ego.y +ego.x >-141.1 and ego.x>-100 and ego.y -ego.x <141.1 and ego.y <100 and ego.y +ego.x < 141.1 and ego.x<100 and ego.y -ego.x >-141.1):
            output.craft_mode = CraftMode.Rendezvous
            output.x = -100

    if ego.craft_mode == CraftMode.Rendezvous:
        if (120<= ego.t ):
            output.craft_mode = CraftMode.Aborting
    
    assert (ego.craft_mode!=CraftMode.Rendezvous or
         ego.x>=-100 and ego.y>=0.57735026919*ego.x and -ego.y>=0.57735026919*ego.x), "Line-of-sight"
    #len = 0.04209517756
    #dist is 0.05081337428
    assert (ego.craft_mode != CraftMode.Rendezvous or \
          ego.vx <= 3.049 and ego.vx >= -3.049 and ego.vy <= 3.049 and ego.vy >= -3.049 and ego.vy <= (-ego.vx + 4.31) and ego.vy <= (ego.vx + 4.31) and ego.vy >= (-ego.vx - 4.31) and ego.vy >= (ego.vx - 4.31)) , "velocity constraint"


    assert (ego.craft_mode!=CraftMode.Aborting or\
         (ego.x<=-.2 or ego.x>=.2 or ego.y<=-.2 or ego.y>=.2)), "Collision avoidance"
    return output
