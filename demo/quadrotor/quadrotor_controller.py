from enum import Enum, auto
import copy


class CraftMode(Enum):
    Follow_Waypoint = auto()


class State:
    x = 0.0
    y = 0.0
    z = 0.0
    vx = 0.0
    vy = 0.0
    vz = 0.0
    craft_mode: CraftMode = CraftMode.Follow_Waypoint
    waypoint_index: int = 0
    done_flag = 0.0  # indicate if the quad rotor reach the waypoint

    def __init__(self, x, y, z, vx, vy, vz, waypoint_index, done_flag, craft_mode):
        pass


def controller(ego: State):
    output = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.Follow_Waypoint:
        if ego.waypoint_index == 0 and ego.done_flag > 0:
            output.craft_mode = CraftMode.Follow_Waypoint
            output.waypoint_index = 1
            output.done_flag = 0
        if ego.waypoint_index == 1 and ego.done_flag > 0:
            output.craft_mode = CraftMode.Follow_Waypoint
            output.waypoint_index = 4
            output.done_flag = 0
        if ego.waypoint_index == 1 and ego.done_flag > 0:
            output.craft_mode = CraftMode.Follow_Waypoint
            output.waypoint_index = 2
            output.done_flag = 0
        if ego.waypoint_index == 2 and ego.done_flag > 0:
            output.craft_mode = CraftMode.Follow_Waypoint
            output.waypoint_index = 3
            output.done_flag = 0
        if ego.waypoint_index == 4 and ego.done_flag > 0:
            output.craft_mode = CraftMode.Follow_Waypoint
            output.waypoint_index = 5
            output.done_flag = 0
    return output
