from typing import Dict
from verse.map.lane import Lane
from verse.map.lane_map import LaneMap
from verse.map.lane_segment import AbstractLane, CircularLane, StraightLane
import math


class Intersection(LaneMap):
    """
    Cross intersection example.

    width: width of each lane
    size: distance from center (of intersection) to when curves and straight lines meet
    length: length of straight road outside of intersection
    """
    def __init__(self, width: float = 3, size: float = 20, length: float = 300):
        assert width > 0 and length > 0
        super().__init__()
        self.width, self.size, self.length = width, size, length
        off = width / 2
        outer_edge = size + length
        segs: Dict[str, AbstractLane] = {
            # Outer straights
            "NR": StraightLane("NR", (-off, outer_edge), (-off, size), width),
            "SR": StraightLane("SR", (off, -outer_edge), (off, -size), width),
            "ER": StraightLane("ER", (outer_edge, off), (size, off), width),
            "WR": StraightLane("WR", (-outer_edge, -off), (-size, -off), width),
            "NL": StraightLane("NL", (off, size), (off, outer_edge), width),
            "EL": StraightLane("EL", (size, -off), (outer_edge, -off), width),
            "SL": StraightLane("SL", (-off, -size), (-off, -outer_edge), width),
            "WL": StraightLane("WL", (-size, off), (-outer_edge, off), width),
            # Inner straights
            "NSR": StraightLane("NSR", (-off, size), (-off, -size), width),
            "NSL": StraightLane("NSL", (off, -size), (off, size), width),
            "WER": StraightLane("WER", (-size, -off), (size, -off), width),
            "WEL": StraightLane("WEL", (size, off), (-size, off), width),
        }
        curve_params = {
            "NW": ((-size, size), 3),
            "NE": ((size, size), 2),
            "SW": ((-size, -size), 0),
            "SE": ((size, -size), 1),
        }
        segs.update(
            {
                (n := k + "IO"[io]): CircularLane(
                    n,
                    center,
                    size + width / 2 * [-1, 1][io],
                    (start + (int(io == 0))) * math.pi / 2,     # XXX inner circulars are clockwise, thus start larger then end
                    (start + (int(io == 1))) * math.pi / 2,     # also python bool converts to int like C would
                    clockwise=io == 0,
                    width=width,
                )
                for k, (center, start) in curve_params.items()
                for io in range(2)
            }
        )
        # straight
        lanes = [
            ("NS", "NR", "NSR", "SL"),
            ("SN", "SR", "NSL", "NL"),
            ("WE", "WR", "WER", "EL"),
            ("EW", "ER", "WEL", "WL"),
        ]
        self.add_lanes([Lane(lane[0], [segs[seg] for seg in lane[1:]]) for lane in lanes])
        # curve
        curves = ["NWI", "NEO", "WNO", "ENI", "SWO", "SEI", "WSI", "ESO"]
        self.add_lanes([Lane(s + e, [segs[s + "R"], segs.get(s + e + d, None) or segs[e + s + d], segs[e + "L"]])
                        for s, e, d in curves])
        all_lanes = [straight[0] for straight in lanes] + [curve[:2] for curve in curves]
        actions = ["Brake", "Accel"]
        self.h_dict = {(lane, actions[i], actions[1 - i]): lane for lane in all_lanes for i in range(2)}
