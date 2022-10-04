from enum import Enum, auto
import copy


class CraftMode(Enum):
    Normal = auto()
    Switch_Down = auto()
    Switch_Up = auto()
    Switch_Left = auto()
    Switch_Right = auto()


class LaneMode(Enum):
    Lane0 = auto()
    Lane1 = auto()
    Lane2 = auto()
    Lane3 = auto()
    Lane4 = auto()


class State:
    x = 0.0
    y = 0.0
    z = 0.0
    vx = 0.0
    vy = 0.0
    vz = 0.0
    craft_mode: CraftMode = CraftMode.Normal
    lane_mode: LaneMode = LaneMode.Lane0

    def __init__(self, x, y, z, vx, vy, vz, craft_mode, lane_mode):
        pass


def controller(ego: State, lane_map):
    output = copy.deepcopy(ego)

    # if ego.craft_mode == CraftMode.Normal:
    #     if lane_map.get_longitudinal_position(others.lane_mode, [others.x, others.y, others.z]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x, ego.y, ego.z]) > 3 \
    #         and lane_map.get_longitudinal_position(others.lane_mode, [others.x, others.y, others.z]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x, ego.y, ego.z]) < 5 \
    #             and ego.lane_mode == others.lane_mode:
    #         if lane_map.has_right(ego.lane_mode):
    #             output.craft_mode = CraftMode.Switch_Right
    #             output.lane_mode = lane_map.right_lane(ego.lane_mode)
    #     if lane_map.get_longitudinal_position(others.lane_mode, [others.x, others.y, others.z]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x, ego.y, ego.z]) > 3 \
    #         and lane_map.get_longitudinal_position(others.lane_mode, [others.x, others.y, others.z]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x, ego.y, ego.z]) < 5 \
    #             and ego.lane_mode == others.lane_mode:
    #         if lane_map.has_left(ego.lane_mode):
    #             output.craft_mode = CraftMode.Switch_Left
    #             output.lane_mode = lane_map.left_lane(ego.lane_mode)
    # if lane_map.get_longitudinal_position(others.lane_mode, [others.x, others.y, others.z]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x, ego.y, ego.z]) > 3 \
    #     and lane_map.get_longitudinal_position(others.lane_mode, [others.x, others.y, others.z]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x, ego.y, ego.z]) < 5 \
    #         and ego.lane_mode == others.lane_mode:
    #     if lane_map.has_up(ego.lane_mode):
    #         output.craft_mode = CraftMode.Switch_Up
    #         output.lane_mode = lane_map.up_lane(ego.lane_mode)
    # if lane_map.get_longitudinal_position(others.lane_mode, [others.x, others.y, others.z]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x, ego.y, ego.z]) > 3 \
    #     and lane_map.get_longitudinal_position(others.lane_mode, [others.x, others.y, others.z]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x, ego.y, ego.z]) < 5 \
    #         and ego.lane_mode == others.lane_mode:
    #     if lane_map.has_down(ego.lane_mode):
    #         output.craft_mode = CraftMode.Switch_Down
    #         output.lane_mode = lane_map.down_lane(ego.lane_mode)

    # assert not (ego.x > 20 and ego.x < 25), "test"
    return output
