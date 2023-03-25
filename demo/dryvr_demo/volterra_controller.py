from enum import Enum, auto
import copy


class CraftMode(Enum):
    inside = auto()
    outside = auto()


class State:
    x = 0.0
    y = 0.0
    # vx = 0.0
    # vy = 0.0
    # total_time = 0.0
    # cycle_time = 0.0
    craft_mode: CraftMode = CraftMode.outside

    def __init__(self, x, y, craft_mode: CraftMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.outside:
        if (((ego.x - 1) ** 2 + (ego.y - 1) ** 2) <= 0.025921):
            output.craft_mode = CraftMode.inside
        # if(ego.x >= .839 and ego.x <= 1.161 and ego.y >= .839 and ego.y<= 1.161 ):
        #     output.craft_mode = CraftMode.inside
    # if ego.craft_mode == CraftMode.inside:
    #     # if (((ego.x - 1) ** 2 + (ego.y - 1) ** 2) >= 0.025921):
    #     #     output.craft_mode = CraftMode.outside
    #     if (ego.x < .839 or ego.x > 1.161 or ego.y < .839 or ego.y > 1.161):
    #         output.craft_mode = CraftMode.outside


    return output
