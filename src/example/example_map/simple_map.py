from src.scene_verifier.map.lane_map import LaneMap
from src.scene_verifier.map.lane_segment import LaneSegment

class SimpleMap(LaneMap):
    def __init__(self):
        super().__init__()
        segment1 = LaneSegment('Lane0', 0)
        segment2 = LaneSegment('Lane1', 3)
        self.add_lanes([segment1,segment2])
        self.left_lane_dict[segment1.id].append(segment2.id)
        self.right_lane_dict[segment2.id].append(segment1.id)

class SimpleMap2(LaneMap):
    def __init__(self):
        super().__init__()
        segment1 = LaneSegment('Lane0', 3)
        segment2 = LaneSegment('Lane1', 0)
        segment3 = LaneSegment('Lane2', -3)
        self.add_lanes([segment1,segment2,segment3])
        self.left_lane_dict[segment2.id].append(segment1.id)
        self.left_lane_dict[segment3.id].append(segment2.id)
        self.right_lane_dict[segment1.id].append(segment2.id)
        self.right_lane_dict[segment2.id].append(segment3.id)

class SimpleMap3(LaneMap):
    def __init__(self):
        super().__init__()
        segment1 = LaneSegment('Lane0', ((0,-30),33))
        segment2 = LaneSegment('Lane1', ((0,-30),30))
        segment3 = LaneSegment('Lane3', ((0,-30),27))
        self.add_lanes([segment1,segment2,segment3])
        self.left_lane_dict[segment2.id].append(segment1.id)
        self.left_lane_dict[segment3.id].append(segment2.id)
        self.right_lane_dict[segment1.id].append(segment2.id)
        self.right_lane_dict[segment2.id].append(segment3.id)

    def get_longitudinal_error(self, lane_idx, agent_state):
        return super().get_longitudinal_error(lane_idx, agent_state)

    def get_lateral_error(self, lane_idx, agent_state):
        return super().get_lateral_error(lane_idx, agent_state) 

    def get_altitude_error(self, lane_idx, agent_state):
        return super().get_altitude_error(lane_idx, agent_state)

if __name__ == "__main__":
    test_map = SimpleMap()
    print(test_map.left_lane_dict)
    print(test_map.right_lane_dict)
    print(test_map.lane_segment_dict)
