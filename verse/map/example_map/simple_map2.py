from verse.map import LaneMap, LaneSegment, StraightLane, CircularLane, Lane

import numpy as np

class SimpleMap2(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane(
            'seg0',
            [0,0],
            [100,0],
            3
        )
        lane0 = Lane('Lane1', [segment0])
        self.add_lanes([lane0])

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
        segment3 = StraightLane(
            'seg3',
            [0,-6],
            [100,-6],
            3
        )
        lane3 = Lane('Lane3', [segment3])
        segment4 = StraightLane(
            'Seg4',
            [0,6],
            [100,6],
            3
        )
        lane4 = Lane('Lane4', [segment4])

        # segment2 = LaneSegment('Lane1', 3)
        # self.add_lanes([segment1,segment2])
        self.add_lanes([lane0, lane1, lane2, lane3, lane4])
        self.left_lane_dict[lane0.id].append(lane4.id)
        self.left_lane_dict[lane1.id].append(lane0.id)
        self.left_lane_dict[lane2.id].append(lane1.id)
        self.left_lane_dict[lane3.id].append(lane2.id)
        self.right_lane_dict[lane4.id].append(lane0.id)
        self.right_lane_dict[lane0.id].append(lane1.id)
        self.right_lane_dict[lane1.id].append(lane2.id)
        self.right_lane_dict[lane2.id].append(lane3.id)

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
            [50,13],
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
            [50,10],
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
            [50,7],
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
            [22.5,3],
            3
        )
        segment1 = CircularLane(
            'Seg1',
            [22.5,13],
            10,
            np.pi*3/2,
            np.pi*2,
            False,
            3
        )
        segment2 = StraightLane(
            'Seg2',
            [32.5,13], 
            [32.5,100],
            3
        )
        lane0 = Lane('Lane0', [segment0, segment1, segment2])
        segment0 = StraightLane(
            'seg0',
            [0,0],
            [22.5,0],
            3
        )
        segment1 = CircularLane(
            'seg1',
            [22.5,13],
            13,
            3*np.pi/2,
            2*np.pi,
            False,
            3
        )
        segment2 = StraightLane(
            'seg2',
            [35.5,13],
            [35.5,100],
            3
        )
        lane1 = Lane('Lane1', [segment0, segment1, segment2])
        segment0 = StraightLane(
            'seg0',
            [0,-3],
            [22.5,-3],
            3
        )
        segment1 = CircularLane(
            'seg1',
            [22.5,13],
            16,
            np.pi*3/2,
            np.pi*2,
            False,
            3
        )
        segment2 = StraightLane(
            'seg2',
            [38.5,13],
            [38.5,100],
            3
        )
        lane2 = Lane('Lane2', [segment0, segment1, segment2])
        self.add_lanes([lane0, lane1, lane2])
        self.left_lane_dict[lane1.id].append(lane0.id)
        self.left_lane_dict[lane2.id].append(lane1.id)
        self.right_lane_dict[lane0.id].append(lane1.id)
        self.right_lane_dict[lane1.id].append(lane2.id)

if __name__ == "__main__":
    test_map = SimpleMap3()
    print(test_map.left_lane_dict)
    print(test_map.right_lane_dict)
    print(test_map.lane_dict)
