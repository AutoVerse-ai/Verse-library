from verse.map.lane_map import LaneMap
from verse.map.lane_segment import StraightLane
from verse.map.lane import Lane
from verse.map.lane_map_3d import LaneMap_3d
from verse.map.lane_segment_3d import StraightLane_3d
from verse.map.lane_3d import Lane_3d


class M1(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane("Seg0", [0, 0], [500, 0], 3)
        lane0 = Lane("T0", [segment0])
        # segment2 = LaneSegment('Lane1', 3)
        # self.add_lanes([segment1,segment2])
        self.add_lanes([lane0])
        self.h_dict = {("T0", "Normal", "Brake"): "T0"}


class M2(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane("Seg0", [0, 0], [500, 0], 3)
        lane0 = Lane("T0", [segment0])
        segment1 = StraightLane("Seg0", [0, 3], [500, 3], 3)
        lane1 = Lane("T1", [segment1])
        # segment2 = LaneSegment('Lane1', 3)
        # self.add_lanes([segment1,segment2])
        self.add_lanes([lane0, lane1])
        self.h_dict = {
            ("T0", "Normal", "Brake"): "T0",
            ("T1", "Normal", "Brake"): "T1",
        }


class M3(LaneMap_3d):
    def __init__(self):
        super().__init__()
        width = 4
        y_offset = 0
        z_offset = 0
        segment0 = StraightLane_3d("seg0", [0, 0, 0], [100, 0 + y_offset, 0 + z_offset], width)
        lane0 = Lane_3d("T0", [segment0])

        segment1 = StraightLane_3d("seg1", [0, 0, 0], [100, 0 + y_offset, 50 + z_offset], width)
        lane1 = Lane_3d("TAvoidUp", [segment1], plotted=False)

        self.add_lanes([lane0, lane1])

        self.h_dict = {
            ("T0", "Normal", "AvoidUp"): "TAvoidUp",
        }

    def h(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> str:
        return self.h_dict[(lane_idx, agent_mode_src, agent_mode_dest)]

    def h_exist(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> bool:
        if (lane_idx, agent_mode_src, agent_mode_dest) in self.h_dict:
            return True
        else:
            return False

    def trans_func(self, lane_idx: str) -> str:
        return lane_idx


class M4(LaneMap_3d):
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
        return self.h_dict[(lane_idx, agent_mode_src, agent_mode_dest)]

    def h_exist(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> bool:
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


class M5(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane("Seg0", [0, 0], [500, 0], 3)
        lane0 = Lane("T0", [segment0])
        segment0 = StraightLane("Seg0", [0, 3], [500, 3], 3)
        lane1 = Lane("T1", [segment0])
        # segment2 = LaneSegment('Lane1', 3)
        # self.add_lanes([segment1,segment2])
        self.add_lanes([lane0, lane1])
        self.h_dict = {
            ("T0", "Normal", "SwitchLeft"): "M01",
            ("T1", "Normal", "SwitchRight"): "M10",
            ("M01", "SwitchLeft", "Normal"): "T1",
            ("M10", "SwitchRight", "Normal"): "T0",
        }
