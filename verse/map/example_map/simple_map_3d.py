from verse.map.lane_map_3d import LaneMap_3d
from verse.map.lane_segment_3d import StraightLane_3d, CircularLane_3d_v1, CircularLane_3d_v2
from verse.map.lane_3d import Lane_3d
from math import pi, tan, sin
from enum import Enum, auto
import numpy as np


class SimpleMap1(LaneMap_3d):
    def __init__(self, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(box_side=box_side, t_v_pair=t_v_pair)
        width = 0.3
        y_offset = 0
        z_offset = 0
        segment0 = StraightLane_3d(
            'seg0',
            [0, 0, 0],
            [100, 0+y_offset, 0+z_offset],
            width
        )
        lane0 = Lane_3d('Lane0', [segment0])
        segment1 = StraightLane_3d(
            'seg0',
            [0, -2*width, 0],
            [100, -2*width+y_offset, 0+z_offset],
            width
        )
        lane1 = Lane_3d('Lane1', [segment1])
        segment2 = StraightLane_3d(
            'seg0',
            [0, 2*width, 0],
            [100, 2*width+y_offset, 0+z_offset],
            width
        )
        lane2 = Lane_3d('Lane2', [segment2])
        segment3 = StraightLane_3d(
            'seg0',
            [0, 0, 2*width],
            [100, 0+y_offset, 2*width+z_offset],
            width
        )
        lane3 = Lane_3d('Lane3', [segment3])
        segment4 = StraightLane_3d(
            'seg0',
            [0, 0, -2*width],
            [100, 0+y_offset, -2*width+z_offset],
            width
        )
        lane4 = Lane_3d('Lane4', [segment4])
        self.add_lanes([lane0, lane1, lane2, lane3, lane4])
        self.pair_lanes(lane0.id, lane1.id, 'right')
        self.pair_lanes(lane0.id, lane2.id, 'left')
        self.pair_lanes(lane0.id, lane3.id, 'up')
        self.pair_lanes(lane0.id, lane4.id, 'down')


class SimpleMap2(LaneMap_3d):
    def __init__(self, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(box_side=box_side, t_v_pair=t_v_pair)
        segment0 = CircularLane_3d_v1(
            'seg0',
            [0, 0, 0],
            6,
            [1, 1, 1],
            0, 2*pi,
            True, 2
        )
        # segment1 = StraightLane_3d(
        #     'seg0',
        #     [0, 3, 0],
        #     [100, 10, 0],
        #     3
        # )
        lane0 = Lane_3d('Lane1', [segment0])
        # lane1 = Lane_3d('Lane2', [segment1])
        # self.add_lanes([lane0, lane1])
        self.add_lanes([lane0])


class SimpleMap3(LaneMap_3d):
    def __init__(self, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(box_side=box_side, t_v_pair=t_v_pair)
        phase = 1.5*pi
        n = np.array([1, 1, 1])
        segment0 = CircularLane_3d_v1(
            'seg0',
            [0, -20, 0],
            20,
            n,
            0, phase,
            True, 5
        )
        start, start_tang, end, end_tang = segment0.get_start_end_tang()
        lens = 40
        end1 = get_end(start, -start_tang, lens)
        segment1 = StraightLane_3d(
            'seg1',
            end1,
            start,
            5
        )
        end2 = get_end(end, end_tang, lens)
        segment2 = StraightLane_3d(
            'seg2',
            end,
            end2,
            5
        )
        segment3 = CircularLane_3d_v2('seg3', end2, end1, -n, phase, True, 5)
        lane0 = Lane_3d('Lane0', [segment0, segment1, segment2, segment3])
        self.add_lanes([lane0])


class SimpleMap4(LaneMap_3d):
    def __init__(self, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(box_side=box_side, t_v_pair=t_v_pair)
        phase = 1.5*pi
        n = np.array([1, 1, 1])
        width = 1
        center0 = np.array([0, -5, 0])
        segment0 = CircularLane_3d_v1(
            'seg0', center0, 5, n, 0, phase, True, width
        )
        start, start_tang, end, end_tang = segment0.get_start_end_tang()
        lens = 10
        end1 = get_end(start, -start_tang, lens)
        segment1 = StraightLane_3d(
            'seg1', end1, start, width
        )
        end2 = get_end(end, end_tang, lens)
        segment2 = StraightLane_3d(
            'seg2', end, end2, width
        )
        segment3 = CircularLane_3d_v2('seg3', end2, end1, -n, phase, True, 1)
        lane0 = Lane_3d('Lane0', [segment0, segment1, segment2, segment3])
        self.add_lanes([lane0])
        center1 = center0+2*width*n
        segment_1_0 = CircularLane_3d_v1(
            'seg0', center1, 5, n, 0, phase, True, width
        )
        start, start_tang, end, end_tang = segment_1_0.get_start_end_tang()
        lens = 10
        end1 = get_end(start, -start_tang, lens)
        segment_1_1 = StraightLane_3d(
            'seg1', end1, start, width
        )
        end2 = get_end(end, end_tang, lens)
        segment_1_2 = StraightLane_3d(
            'seg2', end, end2, width
        )
        segment_1_3 = CircularLane_3d_v2(
            'seg3', end2, end1, -n, phase, True, 1)
        lane1 = Lane_3d(
            'Lane1', [segment_1_0, segment_1_1, segment_1_2, segment_1_3])
        self.add_lanes([lane1])
        self.pair_lanes(lane0.id, lane1.id, 'up')


class SimpleMap5(LaneMap_3d):
    def __init__(self, box_side: dict = {}, t_v_pair: dict = {}):
        super().__init__(box_side=box_side, t_v_pair=t_v_pair)
        phase = 1.5*pi
        n = np.array([1, 1, 1])
        width = 3
        radius = 20
        center0 = np.array([0, -20, 0])
        segment0 = CircularLane_3d_v1(
            'seg0', center0, radius, n, 0, phase, True, width
        )
        start, start_tang, end, end_tang = segment0.get_start_end_tang()
        lens = 2*radius
        end1 = get_end(start, -start_tang, lens)
        segment1 = StraightLane_3d(
            'seg1', end1, start, width
        )
        end2 = get_end(end, end_tang, lens)
        segment2 = StraightLane_3d(
            'seg2', end, end2, width
        )
        segment3 = CircularLane_3d_v2('seg3', end2, end1, -n, phase, True, 1)
        lane0 = Lane_3d('Lane0', [segment0, segment1, segment2, segment3])
        self.add_lanes([lane0])
        center1 = center0+2*width*n
        segment_1_0 = CircularLane_3d_v1(
            'seg0', center1, radius, n, 0, phase, True, width
        )
        start, start_tang, end, end_tang = segment_1_0.get_start_end_tang()
        lens = 2*radius
        end1 = get_end(start, -start_tang, lens)
        segment_1_1 = StraightLane_3d(
            'seg1', end1, start, width
        )
        end2 = get_end(end, end_tang, lens)
        segment_1_2 = StraightLane_3d(
            'seg2', end, end2, width
        )
        segment_1_3 = CircularLane_3d_v2(
            'seg3', end2, end1, -n, phase, True, 1)
        lane1 = Lane_3d(
            'Lane1', [segment_1_0, segment_1_1, segment_1_2, segment_1_3])
        self.add_lanes([lane1])
        self.pair_lanes(lane0.id, lane1.id, 'up')


class SimpleMap6(LaneMap_3d):
    def __init__(self):
        super().__init__()
        phase = 1.5*pi
        n = np.array([0, 0, 1])
        width = 2
        radius = 10
        center1 = np.array([0, 0, 0])
        segment_1_0 = CircularLane_3d_v1(
            'seg0', center1, radius, n, 0, phase, True, width
        )
        start, start_tang, end, end_tang = segment_1_0.get_start_end_tang()
        lens = 2*radius
        end1 = get_end(start, -start_tang, lens)
        segment_1_1 = StraightLane_3d(
            'seg1', end1, start, width
        )
        end2 = get_end(end, end_tang, lens)
        segment_1_2 = StraightLane_3d(
            'seg2', end, end2, width
        )
        segment_1_3 = CircularLane_3d_v2(
            'seg3', end2, end1, -n, phase, True, width)
        lane1 = Lane_3d('T1', [segment_1_0, segment_1_1,
                        segment_1_2, segment_1_3])

        center0 = center1+2*width*n
        segment_0_0 = CircularLane_3d_v1(
            'seg0', center0, radius, n, 0, phase, True, width
        )
        start, start_tang, end, end_tang = segment_0_0.get_start_end_tang()
        lens = 2*radius
        end1 = get_end(start, -start_tang, lens)
        segment_0_1 = StraightLane_3d(
            'seg1', end1, start, width
        )
        end2 = get_end(end, end_tang, lens)
        segment_0_2 = StraightLane_3d(
            'seg2', end, end2, width
        )
        segment_0_3 = CircularLane_3d_v2(
            'seg3', end2, end1, -n, phase, True, width)
        lane0 = Lane_3d(
            'T0', [segment_0_0, segment_0_1, segment_0_2, segment_0_3])

        center2 = center1-2*width*n
        segment_2_0 = CircularLane_3d_v1(
            'seg0', center2, radius, n, 0, phase, True, width
        )
        start, start_tang, end, end_tang = segment_2_0.get_start_end_tang()
        lens = 2*radius
        end1 = get_end(start, -start_tang, lens)
        segment_2_1 = StraightLane_3d(
            'seg1', end1, start, width
        )
        end2 = get_end(end, end_tang, lens)
        segment_2_2 = StraightLane_3d(
            'seg2', end, end2, width
        )
        segment_2_3 = CircularLane_3d_v2(
            'seg3', end2, end1, -n, phase, True, width)
        lane2 = Lane_3d(
            'T2', [segment_2_0, segment_2_1, segment_2_2, segment_2_3])

        self.add_lanes([lane0, lane1, lane2])
        self.pair_lanes(lane1.id, lane0.id, 'up')
        self.pair_lanes(lane1.id, lane2.id, 'down')

        self.h_dict = {
            ('T1', 'Normal', 'MoveUp'): 'M10',
            ('T0', 'Normal', 'MoveDown'): 'M01',
            ('T1', 'Normal', 'MoveDown'): 'M12',
            ('T2', 'Normal', 'MoveUp'): 'M21',
            ('M10', 'MoveUp', 'Normal'): 'T0',
            ('M01', 'MoveDown', 'Normal'): 'T1',
            ('M12', 'MoveDown', 'Normal'): 'T2',
            ('M21', 'MoveUp', 'Normal'): 'T1',
        }

    def h(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> str:
        return self.h_dict[(lane_idx, agent_mode_src, agent_mode_dest)]

    def h_exist(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> bool:
        if (lane_idx, agent_mode_src, agent_mode_dest) in self.h_dict:
            return True
        else:
            return False

    def trans_func(self, lane_idx: str) -> str:
        lane = ''
        if lane_idx[-1] == '0':
            lane = 'T0'
        elif lane_idx[-1] == '1':
            lane = 'T1'
        elif lane_idx[-1] == '2':
            lane = 'T2'
        return lane

class SimpleMap7(LaneMap_3d):

    def __init__(self):
        super().__init__()
        width = 4
        y_offset = 0
        z_offset = 0
        segment0 = StraightLane_3d(
            'seg0',
            [0, 0, 0],
            [100, 0+y_offset, 0+z_offset],
            width
        )
        lane0 = Lane_3d('T1', [segment0])
        segment3 = StraightLane_3d(
            'seg0',
            [0, 0, 2*width],
            [100, 0+y_offset, 2*width+z_offset],
            width
        )
        lane3 = Lane_3d('T0', [segment3])
        segment4 = StraightLane_3d(
            'seg0',
            [0, 0, -2*width],
            [100, 0+y_offset, -2*width+z_offset],
            width
        )
        lane4 = Lane_3d('T2', [segment4])
        self.add_lanes([lane0, lane3, lane4])

        self.h_dict = {
            ('T1', 'Normal', 'MoveUp'): 'M10',
            ('T0', 'Normal', 'MoveDown'): 'M01',
            ('T1', 'Normal', 'MoveDown'): 'M12',
            ('T2', 'Normal', 'MoveUp'): 'M21',
            ('M10', 'MoveUp', 'Normal'): 'T0',
            ('M01', 'MoveDown', 'Normal'): 'T1',
            ('M12', 'MoveDown', 'Normal'): 'T2',
            ('M21', 'MoveUp', 'Normal'): 'T1',
        }

    def h(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> str:
        return self.h_dict[(lane_idx, agent_mode_src, agent_mode_dest)]

    def h_exist(self, lane_idx: str, agent_mode_src: str, agent_mode_dest: str) -> bool:
        if (lane_idx, agent_mode_src, agent_mode_dest) in self.h_dict:
            return True
        else:
            return False

    def trans_func(self, lane_idx: str) -> str:
        lane = ''
        if lane_idx[-1] == '0':
            lane = 'T0'
        elif lane_idx[-1] == '1':
            lane = 'T1'
        elif lane_idx[-1] == '2':
            lane = 'T2'
        return lane

def get_end(start, n, lens):
    n = n/np.linalg.norm(n)
    return (start+n*lens).tolist()


def get_center_radius(start, end, n, phase):
    n, end, start = np.array(n), np.array(end), np.array(start)
    assert (0 < phase < 2*pi and np.all(start != end))
    l_n = np.cross(n, end-start)
    l_n = l_n/np.linalg.norm(l_n)
    mid = (end+start)/2
    l = np.linalg.norm(end-start)/2
    if phase > pi:
        theta = (phase-pi)/2
    elif phase < pi:
        theta = phase/2
    else:
        return mid, l/2
    return mid+l/tan(theta)*l_n, l/sin(theta)
