from enum import Enum, auto
import copy


class CraftMode(Enum):
    inside = auto()
    outside = auto()


class State:
    x = 0.0
    y = 0.0
    t_loc = 0.0
    craft_mode: CraftMode = CraftMode.outside

    def __init__(self, x, y, craft_mode: CraftMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.outside:
        # if (((ego.x - 1) ** 2 + (ego.y - 1) ** 2) <= 0.025921):
        #     output.craft_mode = CraftMode.inside
        # if(ego.x >= .839 and ego.x <= 1.161 and ego.y >= .839 and ego.y<= 1.161 ):
        #     output.craft_mode = CraftMode.inside
        # if ego.x < 1+0.161 and ego.x>1-0.161 and ego.y<1+0.161 and ego.y>1-0.161 and ego.t_loc > 1.0:
        if ego.x<1+0.161 and ego.x>1-0.161 and ego.y<1+0.161 and ego.y>1-0.161\
            and ego.y<-ego.x+ 2.23  and ego.y>ego.x- 0.2276899  \
            and ego.y<ego.x+0.227669 and ego.y>-ego.x +1.77 and ego.t_loc>0.5:
            output.craft_mode = CraftMode.inside
            # if ego.x <1+0.161 and ego.y > 1:
            #     output.x = 1+0.161
            output.t_loc = 0
            # output.x = 1.05
    if ego.craft_mode == CraftMode.inside:
        # if (((ego.x - 1) ** 2 + (ego.y - 1) ** 2) >= 0.025921):
        #     output.craft_mode = CraftMode.outside
        # if (ego.x < .839 or ego.x > 1.161 or ego.y < .839 or ego.y > 1.161):
        #     output.craft_mode = CraftMode.outside
        if not (ego.x<1+0.161 and ego.x>1-0.161 and ego.y<1+0.161 and ego.y>1-0.161\
            and ego.y<-ego.x+ 2.23  and ego.y>ego.x- 0.2276899  \
            and ego.y<ego.x+0.227669 and ego.y>-ego.x +1.77) and ego.t_loc>0.5:
            output.craft_mode = CraftMode.outside
            output.t_loc = 0
            # output.x = 1.05


    return output
