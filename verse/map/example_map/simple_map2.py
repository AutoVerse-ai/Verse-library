from typing import Optional
from verse.map import LaneMap, LaneSegment, StraightLane, CircularLane, Lane

import numpy as np


class SimpleMap2(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane("seg0", [0, 0], [100, 0], 3)
        lane0 = Lane("Lane1", [segment0])
        self.add_lanes([lane0])


class SimpleMap3(LaneMap):
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
            ("M01", "SwitchRight", "Normal"): "T1",
            ("M12", "SwitchRight", "Normal"): "T2",
            ("M10", "SwitchLeft", "Normal"): "T0",
            ("M21", "SwitchLeft", "Normal"): "T1",
        }


class SimpleMap4(LaneMap):
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
        return self.left_dict[lane_mode]

    def right_lane(self, lane_mode):
        return self.right_dict[lane_mode]


class SimpleMap4Switch2(LaneMap):
    def __init__(self):
        super().__init__()
        self.add_lanes(
            [
                Lane(f"T{i}", [StraightLane(f"seg{i}", [0, 6 - 3 * i], [100, 6 - 3 * i], 3)])
                for i in range(5)
            ]
        )

    def h(self, lane_idx: str, mode_src: str, mode_dest: str) -> Optional[str]:
        src_sw, dst_sw = mode_src.startswith("Switch"), mode_dest.startswith("Switch")
        if src_sw:
            if dst_sw:
                return None
            else:
                return "T" + lane_idx[2]
        else:
            if dst_sw:
                typ = mode_dest[6:]
                ind = int(lane_idx[1])
                if typ == "Left" and ind > 0:
                    new_ind = ind - 1
                if typ == "Left2" and ind > 1:
                    new_ind = ind - 2
                elif typ == "Right" and ind < 4:
                    new_ind = ind + 1
                elif typ == "Right2" and ind < 3:
                    new_ind = ind + 2
                else:
                    return None
                return f"M{ind}{new_ind}"
            else:
                return lane_idx


class SimpleMap5(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane("Seg0", [0, 3], [15, 3], 3)
        segment1 = StraightLane("Seg1", [15, 3], [25, 13], 3)
        segment2 = StraightLane("Seg2", [25, 13], [50, 13], 3)
        lane0 = Lane("Lane0", [segment0, segment1, segment2])
        segment0 = StraightLane("seg0", [0, 0], [17, 0], 3)
        segment1 = StraightLane("seg1", [17, 0], [27, 10], 3)
        segment2 = StraightLane("seg2", [27, 10], [50, 10], 3)
        lane1 = Lane("Lane1", [segment0, segment1, segment2])
        segment0 = StraightLane("seg0", [0, -3], [19, -3], 3)
        segment1 = StraightLane("seg1", [19, -3], [29, 7], 3)
        segment2 = StraightLane("seg2", [29, 7], [50, 7], 3)
        lane2 = Lane("Lane2", [segment0, segment1, segment2])
        self.add_lanes([lane0, lane1, lane2])
        self.left_lane_dict[lane1.id].append(lane0.id)
        self.left_lane_dict[lane2.id].append(lane1.id)
        self.right_lane_dict[lane0.id].append(lane1.id)
        self.right_lane_dict[lane1.id].append(lane2.id)


class SimpleMap6(LaneMap):
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
        # self.left_lane_dict[lane1.id].append(lane0.id)
        # self.left_lane_dict[lane2.id].append(lane1.id)
        # self.right_lane_dict[lane0.id].append(lane1.id)
        # self.right_lane_dict[lane1.id].append(lane2.id)


if __name__ == "__main__":
    test_map = SimpleMap3()
    print(test_map.left_lane_dict)
    print(test_map.right_lane_dict)
    print(test_map.lane_dict)
