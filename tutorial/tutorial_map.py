from verse.map.lane_map import LaneMap
from verse.map.lane_segment import StraightLane
from verse.map.lane import Lane

class M1(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane(
            'Seg0',
            [0,0],
            [500,0],
            3
        )
        lane0 = Lane('T0', [segment0])
        # segment2 = LaneSegment('Lane1', 3)
        # self.add_lanes([segment1,segment2])
        self.add_lanes([lane0])
        self.h_dict = {("T0","Normal","Brake"):"T0"}

class M2(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane(
            'Seg0',
            [0,0],
            [500,0],
            3
        )
        lane0 = Lane('T0', [segment0])
        segment1 = StraightLane(
            'Seg0',
            [0,3],
            [500,3],
            3
        )
        lane1 = Lane('T1', [segment1])
        # segment2 = LaneSegment('Lane1', 3)
        # self.add_lanes([segment1,segment2])
        self.add_lanes([lane0, lane1])
        self.h_dict = {
            ("T0","Normal","Brake"):"T0",
            ("T1","Normal","Brake"):"T1",
        }
