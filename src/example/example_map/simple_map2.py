from src.scene_verifier.map.lane_map import LaneMap
from src.scene_verifier.map.lane_segment import LaneSegment, StraightLane, CircularLane
from src.scene_verifier.map.lane import Lane

import numpy as np

class SimpleMap3(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane(
            'Seg0',
            [0,3],
            [100,3],
            3
        )
        lane0 = Lane('Lane0', [segment0])
        segment1 = StraightLane(
            'seg0',
            [0,0],
            [100,0],
            3
        )
        lane1 = Lane('Lane1', [segment1])
        segment2 = StraightLane(
            'seg0',
            [0,-3],
            [100,-3],
            3
        )
        lane2 = Lane('Lane2', [segment2])
        # segment2 = LaneSegment('Lane1', 3)
        # self.add_lanes([segment1,segment2])
        self.add_lanes([lane0, lane1, lane2])
        self.left_lane_dict[lane1.id].append(lane0.id)
        self.left_lane_dict[lane2.id].append(lane1.id)
        self.right_lane_dict[lane0.id].append(lane1.id)
        self.right_lane_dict[lane1.id].append(lane2.id)

class SimpleMap4(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane(
            'Seg0',
            [0,3],
            [30,3],
            3
        )
        segment1 = StraightLane(
            'Seg1',
            [30,3], 
            [50,23],
            3
        )
        lane0 = Lane('Lane0', [segment0, segment1])
        segment0 = StraightLane(
            'seg0',
            [0,0],
            [33,0],
            3
        )
        segment1 = StraightLane(
            'seg0',
            [33,0],
            [53,20],
            3
        )
        lane1 = Lane('Lane1', [segment0, segment1])
        segment0 = StraightLane(
            'seg0',
            [0,-3],
            [36,-3],
            3
        )
        segment1 = StraightLane(
            'seg0',
            [36,-3],
            [56,17],
            3
        )
        lane2 = Lane('Lane2', [segment0, segment1])
        self.add_lanes([lane0, lane1, lane2])
        self.left_lane_dict[lane1.id].append(lane0.id)
        self.left_lane_dict[lane2.id].append(lane1.id)
        self.right_lane_dict[lane0.id].append(lane1.id)
        self.right_lane_dict[lane1.id].append(lane2.id)

class SimpleMap5(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane(
            'Seg0',
            [0,3],
            [15,3],
            3
        )
        segment1 = StraightLane(
            'Seg1',
            [15,3], 
            [25,13],
            3
        )
        segment2 = StraightLane(
            'Seg2',
            [25,13], 
            [100,13],
            3
        )
        lane0 = Lane('Lane0', [segment0, segment1, segment2])
        segment0 = StraightLane(
            'seg0',
            [0,0],
            [17,0],
            3
        )
        segment1 = StraightLane(
            'seg1',
            [17,0],
            [27,10],
            3
        )
        segment2 = StraightLane(
            'seg2',
            [27,10],
            [100,10],
            3
        )
        lane1 = Lane('Lane1', [segment0, segment1, segment2])
        segment0 = StraightLane(
            'seg0',
            [0,-3],
            [19,-3],
            3
        )
        segment1 = StraightLane(
            'seg1',
            [19,-3],
            [29,7],
            3
        )
        segment2 = StraightLane(
            'seg2',
            [29,7],
            [100,7],
            3
        )
        lane2 = Lane('Lane2', [segment0, segment1, segment2])
        self.add_lanes([lane0, lane1, lane2])
        self.left_lane_dict[lane1.id].append(lane0.id)
        self.left_lane_dict[lane2.id].append(lane1.id)
        self.right_lane_dict[lane0.id].append(lane1.id)
        self.right_lane_dict[lane1.id].append(lane2.id)

class SimpleMap6(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane(
            'Seg0',
            [0,3],
            [15,3],
            3
        )
        segment1 = CircularLane(
            'Seg1',
            [15,8],
            5,
            np.pi*3/2,
            np.pi*2,
            False,
            3
        )
        segment2 = StraightLane(
            'Seg2',
            [20,8], 
            [20,100],
            3
        )
        lane0 = Lane('Lane0', [segment0, segment1, segment2])
        segment0 = StraightLane(
            'seg0',
            [0,0],
            [18,0],
            3
        )
        segment1 = CircularLane(
            'seg1',
            [18,5],
            5,
            3*np.pi/2,
            2*np.pi,
            False,
            3
        )
        segment2 = StraightLane(
            'seg2',
            [23,5],
            [23,100],
            3
        )
        lane1 = Lane('Lane1', [segment0, segment1, segment2])
        segment0 = StraightLane(
            'seg0',
            [0,-3],
            [21,-3],
            3
        )
        segment1 = CircularLane(
            'seg1',
            [21,2],
            5,
            np.pi*3/2,
            np.pi*2,
            False,
            3
        )
        segment2 = StraightLane(
            'seg2',
            [26,2],
            [26,100],
            3
        )
        lane2 = Lane('Lane2', [segment0, segment1, segment2])
        self.add_lanes([lane0, lane1, lane2])
        self.left_lane_dict[lane1.id].append(lane0.id)
        self.left_lane_dict[lane2.id].append(lane1.id)
        self.right_lane_dict[lane0.id].append(lane1.id)
        self.right_lane_dict[lane1.id].append(lane2.id)


class SimpleMap7(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = CircularLane(
            'Seg0',
            [0,5],
            5,
            np.pi*3/2,
            np.pi*2,
            False,
            3
        )
        segment1 = StraightLane(
            'Seg1',
            [5,5],
            [5,100],
            width = 3
        )
        lane0 = Lane('Lane0',[segment0,segment1])
        self.add_lanes([lane0])



if __name__ == "__main__":
    test_map = SimpleMap3()
    print(test_map.left_lane_dict)
    print(test_map.right_lane_dict)
    print(test_map.lane_segment_dict)
