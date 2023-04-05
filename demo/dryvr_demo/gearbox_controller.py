from enum import Enum, auto
import copy


class CraftMode(Enum):
    Move = auto()
    Meshed = auto()


class State:
    vx = 0.0
    vy = 0.0
    px = 0.0
    py =0.0
    ii =0.0
    trans = 0.0
    one = 0.0
    craft_mode: CraftMode = CraftMode.Move

    def __init__(self, x, y, craft_mode: CraftMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.Move:
        if(ego.py + 0.726542528005361 * ego.px >0 and 0.587785252292473 * ego.vx + 0.809016994374947 * ego.vy >0 and ego.trans < 8):
            output.ii = ego.ii + 7.74867751838096*ego.vx + 10.66513964386099*ego.vy
            output.vx = -0.42329949064832*ego.vx - 1.95900368634417*ego.vy
            output.vy = -0.346343193165*ego.vx + 0.52329949064*ego.vy
            output.trans = ego.trans + ego.one
        if(ego.py - 0.726542528005361*ego.px<0  and  0.587785252292473*ego.vx - 0.809016994374947*ego.vy>0 and  ego.trans<8):
            output.ii = ego.ii + 7.74867751838096*ego.vx - 10.66513964386099*ego.vy
            output.vx = -0.42329949064832*ego.vx + 1.95900368634417*ego.vy
            output.vy = 0.346343193165*ego.vx + 0.52329949064*ego.vy
            output.trans = ego.trans + ego.one
        if( ego.px >-0.002 and ego.vx>0 and ego.vy<0):
            output.craft_mode = CraftMode.Meshed
            output.ii = ego.ii + 3.2*ego.vx-3.2*ego.vy
            output.vx = 0
            output.vy = 0
            output.trans = ego.trans + ego.one
        if (ego.px > -0.002 and ego.vx < 0 and ego.vy > 0):
            output.craft_mode = CraftMode.Meshed
            output.ii = ego.ii + 3.2 * ego.vx - 3.2 * ego.vy
            output.vx = 0
            output.vy = 0
            output.trans = ego.trans + ego.one
        if (ego.px > -0.002 and ego.vx < 0 and ego.vy < 0):
            output.craft_mode = CraftMode.Meshed
            output.ii = ego.ii + 3.2 * ego.vx - 3.2 * ego.vy
            output.vx = 0
            output.vy = 0
            output.trans = ego.trans + ego.one
        if (ego.px > -0.002 and ego.vx > 0 and ego.vy > 0):
            output.craft_mode = CraftMode.Meshed
            output.ii = ego.ii + 3.2 * ego.vx - 3.2 * ego.vy
            output.vx = 0
            output.vy = 0
            output.trans = ego.trans + ego.one








    return output
