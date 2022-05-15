from src.scene_verifier.map.lane_map import LaneMap
from src.scene_verifier.map.lane_segment import LaneSegment, StraightLane
from src.scene_verifier.map.lane import Lane

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


if __name__ == "__main__":
    test_map = SimpleMap3()
    print(test_map.left_lane_dict)
    print(test_map.right_lane_dict)
    print(test_map.lane_segment_dict)
