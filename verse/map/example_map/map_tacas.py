from verse.map.lane_map_3d import LaneMap_3d
from verse.map.lane_segment_3d import StraightLane_3d, CircularLane_3d_v1, CircularLane_3d_v2
from verse.map.lane_3d import Lane_3d
from math import pi
import numpy as np
from verse.map import LaneMap, LaneSegment, StraightLane, CircularLane, Lane
from enum import Enum


class M1(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane("Seg0", [0, 3], [500, 3], 3)
        lane0 = Lane("T0", [segment0])
        segment1 = StraightLane("seg0", [0, 0], [500, 0], 3)
        lane1 = Lane("T1", [segment1])
        segment2 = StraightLane("seg0", [0, -3], [500, -3], 3)
        lane2 = Lane("T2", [segment2])
        # segment2 = LaneSegment('Lane1', 3)
        # self.add_lanes([segment1,segment2])
        self.add_lanes([lane0, lane1, lane2])
        self.h_dict = {
            ("T0", "Normal", "SwitchRight"): "M01",
            ("T1", "Normal", "SwitchRight"): "M12",
            ("T1", "Normal", "SwitchLeft"): "M10",
            ("T2", "Normal", "SwitchLeft"): "M21",
            ("T0", "Normal", "Brake"): "T0",
            ("T1", "Normal", "Brake"): "T1",
            ("T2", "Normal", "Brake"): "T2",
            ("M01", "SwitchRight", "Normal"): "T1",
            ("M12", "SwitchRight", "Normal"): "T2",
            ("M10", "SwitchLeft", "Normal"): "T0",
            ("M21", "SwitchLeft", "Normal"): "T1",
        }


class M2(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane("Seg0", [0, 3], [100, 3], 3)
        lane0 = Lane("T0", [segment0])
        segment1 = StraightLane("seg0", [0, 0], [100, 0], 3)
        lane1 = Lane("T1", [segment1])
        segment2 = StraightLane("seg0", [0, -3], [100, -3], 3)
        lane2 = Lane("T2", [segment2])
        segment3 = StraightLane("seg3", [0, -6], [100, -6], 3)
        lane3 = Lane("T3", [segment3])
        segment4 = StraightLane("Seg4", [0, 6], [100, 6], 3)
        lane4 = Lane("T4", [segment4])

        # segment2 = LaneSegment('Lane1', 3)
        # self.add_lanes([segment1,segment2])
        self.add_lanes([lane0, lane1, lane2, lane3, lane4])
        self.h_dict = {
            ("M04", "SwitchLeft", "Normal"): "T4",
            ("M10", "SwitchLeft", "Normal"): "T0",
            ("M21", "SwitchLeft", "Normal"): "T1",
            ("M32", "SwitchLeft", "Normal"): "T2",
            ("T0", "Normal", "SwitchLeft"): "M04",
            ("T1", "Normal", "SwitchLeft"): "M10",
            ("T2", "Normal", "SwitchLeft"): "M21",
            ("T3", "Normal", "SwitchLeft"): "M32",
            ("T4", "Normal", "SwitchRight"): "M40",
            ("T0", "Normal", "SwitchRight"): "M01",
            ("T1", "Normal", "SwitchRight"): "M12",
            ("T2", "Normal", "SwitchRight"): "M23",
            ("M40", "SwitchRight", "Normal"): "T0",
            ("M01", "SwitchRight", "Normal"): "T1",
            ("M12", "SwitchRight", "Normal"): "T2",
            ("M23", "SwitchRight", "Normal"): "T3",
        }
        self.left_dict = {
            "T0": "T4",
            "T1": "T0",
            "T2": "T1",
            "T3": "T2",
        }
        self.right_dict = {
            "T4": "T0",
            "T0": "T1",
            "T1": "T2",
            "T2": "T3",
        }

    def left_lane(self, lane_mode):
        if isinstance(lane_mode, Enum):
            lane_mode = lane_mode.name
        return self.left_dict[lane_mode]

    def right_lane(self, lane_mode):
        if isinstance(lane_mode, Enum):
            lane_mode = lane_mode.name
        return self.right_dict[lane_mode]


class M3(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane("Seg0", [0, 3], [22.5, 3], 3)
        segment1 = CircularLane("Seg1", [22.5, 13], 10, np.pi * 3 / 2, np.pi * 2, False, 3)
        segment2 = StraightLane("Seg2", [32.5, 13], [32.5, 100], 3)
        lane0 = Lane("T0", [segment0, segment1, segment2])
        segment0 = StraightLane("seg0", [0, 0], [22.5, 0], 3)
        segment1 = CircularLane("seg1", [22.5, 13], 13, 3 * np.pi / 2, 2 * np.pi, False, 3)
        segment2 = StraightLane("seg2", [35.5, 13], [35.5, 100], 3)
        lane1 = Lane("T1", [segment0, segment1, segment2])
        segment0 = StraightLane("seg0", [0, -3], [22.5, -3], 3)
        segment1 = CircularLane("seg1", [22.5, 13], 16, np.pi * 3 / 2, np.pi * 2, False, 3)
        segment2 = StraightLane("seg2", [38.5, 13], [38.5, 100], 3)
        lane2 = Lane("T2", [segment0, segment1, segment2])
        self.add_lanes([lane0, lane1, lane2])
        self.h_dict = {
            ("T0", "Normal", "SwitchRight"): "M01",
            ("T1", "Normal", "SwitchRight"): "M12",
            ("T1", "Normal", "SwitchLeft"): "M10",
            ("T2", "Normal", "SwitchLeft"): "M21",
            ("M01", "SwitchRight", "Normal"): "T1",
            ("M12", "SwitchRight", "Normal"): "T2",
            ("M10", "SwitchLeft", "Normal"): "T0",
            ("M21", "SwitchLeft", "Normal"): "T1",
        }


class M5(LaneMap_3d):
    def __init__(self):
        super().__init__()
        width = 4
        y_offset = 0
        z_offset = 0
        segment0 = StraightLane_3d("seg0", [0, 0, 0], [100, 0 + y_offset, 0 + z_offset], width)
        lane0 = Lane_3d("T1", [segment0])
        segment3 = StraightLane_3d(
            "seg0", [0, 0, 2 * width], [100, 0 + y_offset, 2 * width + z_offset], width
        )
        lane3 = Lane_3d("T0", [segment3])
        segment4 = StraightLane_3d(
            "seg0", [0, 0, -2 * width], [100, 0 + y_offset, -2 * width + z_offset], width
        )
        lane4 = Lane_3d("T2", [segment4])
        self.add_lanes([lane0, lane3, lane4])

        self.h_dict = {
            ("T1", "Normal", "MoveUp"): "M10",
            ("T0", "Normal", "MoveDown"): "M01",
            ("T1", "Normal", "MoveDown"): "M12",
            ("T2", "Normal", "MoveUp"): "M21",
            ("M10", "MoveUp", "Normal"): "T0",
            ("M01", "MoveDown", "Normal"): "T1",
            ("M12", "MoveDown", "Normal"): "T2",
            ("M21", "MoveUp", "Normal"): "T1",
        }

    def h(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> str:
        if not isinstance(lane_idx, str):
            lane_idx = lane_idx.name
        if not isinstance(agent_mode_src, str):
            agent_mode_src = agent_mode_src.name
        if not isinstance(agent_mode_dest, str):
            agent_mode_dest = agent_mode_dest.name
        return self.h_dict[(lane_idx, agent_mode_src, agent_mode_dest)]

    def h_exist(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> bool:
        if not isinstance(lane_idx, str):
            lane_idx = lane_idx.name
        if not isinstance(agent_mode_src, str):
            agent_mode_src = agent_mode_src.name
        if not isinstance(agent_mode_dest, str):
            agent_mode_dest = agent_mode_dest.name
        if (lane_idx, agent_mode_src, agent_mode_dest) in self.h_dict:
            return True
        else:
            return False

    def trans_func(self, lane_idx: str) -> str:
        lane = ""
        if lane_idx[-1] == "0":
            lane = "T0"
        elif lane_idx[-1] == "1":
            lane = "T1"
        elif lane_idx[-1] == "2":
            lane = "T2"
        return lane


def get_end(start, n, lens):
    n = n / np.linalg.norm(n)
    return (start + n * lens).tolist()


class M6(LaneMap_3d):
    def __init__(self):
        super().__init__()
        phase = 1.5 * pi
        n = np.array([0, 0, 1])
        width = 2
        radius = 10
        center1 = np.array([0, 0, 0])
        segment_1_0 = CircularLane_3d_v1("seg0", center1, radius, n, 0, phase, True, width)
        start, start_tang, end, end_tang = segment_1_0.get_start_end_tang()
        lens = 2 * radius
        end1 = get_end(start, -start_tang, lens)
        segment_1_1 = StraightLane_3d("seg1", end1, start, width)
        end2 = get_end(end, end_tang, lens)
        segment_1_2 = StraightLane_3d("seg2", end, end2, width)
        segment_1_3 = CircularLane_3d_v2("seg3", end2, end1, -n, phase, True, width)
        lane1 = Lane_3d("T1", [segment_1_0, segment_1_1, segment_1_2, segment_1_3])

        center0 = center1 + 2 * width * n
        segment_0_0 = CircularLane_3d_v1("seg0", center0, radius, n, 0, phase, True, width)
        start, start_tang, end, end_tang = segment_0_0.get_start_end_tang()
        lens = 2 * radius
        end1 = get_end(start, -start_tang, lens)
        segment_0_1 = StraightLane_3d("seg1", end1, start, width)
        end2 = get_end(end, end_tang, lens)
        segment_0_2 = StraightLane_3d("seg2", end, end2, width)
        segment_0_3 = CircularLane_3d_v2("seg3", end2, end1, -n, phase, True, width)
        lane0 = Lane_3d("T0", [segment_0_0, segment_0_1, segment_0_2, segment_0_3])

        center2 = center1 - 2 * width * n
        segment_2_0 = CircularLane_3d_v1("seg0", center2, radius, n, 0, phase, True, width)
        start, start_tang, end, end_tang = segment_2_0.get_start_end_tang()
        lens = 2 * radius
        end1 = get_end(start, -start_tang, lens)
        segment_2_1 = StraightLane_3d("seg1", end1, start, width)
        end2 = get_end(end, end_tang, lens)
        segment_2_2 = StraightLane_3d("seg2", end, end2, width)
        segment_2_3 = CircularLane_3d_v2("seg3", end2, end1, -n, phase, True, width)
        lane2 = Lane_3d("T2", [segment_2_0, segment_2_1, segment_2_2, segment_2_3])

        self.add_lanes([lane0, lane1, lane2])
        self.pair_lanes(lane1.id, lane0.id, "up")
        self.pair_lanes(lane1.id, lane2.id, "down")

        self.h_dict = {
            ("T1", "Normal", "MoveUp"): "M10",
            ("T0", "Normal", "MoveDown"): "M01",
            ("T1", "Normal", "MoveDown"): "M12",
            ("T2", "Normal", "MoveUp"): "M21",
            ("M10", "MoveUp", "Normal"): "T0",
            ("M01", "MoveDown", "Normal"): "T1",
            ("M12", "MoveDown", "Normal"): "T2",
            ("M21", "MoveUp", "Normal"): "T1",
        }

    def h_func(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> str:
        return self.h_dict[(lane_idx, agent_mode_src, agent_mode_dest)]

    def h_exist_func(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> bool:
        if (lane_idx, agent_mode_src, agent_mode_dest) in self.h_dict:
            return True
        else:
            return False

    def trans_func(self, lane_idx: str) -> str:
        lane = ""
        if lane_idx[-1] == "0":
            lane = "T0"
        elif lane_idx[-1] == "1":
            lane = "T1"
        elif lane_idx[-1] == "2":
            lane = "T2"
        return lane
