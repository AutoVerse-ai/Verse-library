from typing import Dict, Optional

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
    lanes: number of lanes per direction (e.g. north and into the intersection)
    """
    def __init__(self, width: float = 3, size: float = 20, length: float = 300, lanes: int = 2):
        assert width > 0 and lanes > 0 and length > 0
        assert size >= (1 + lanes) * width, "Intersection too small for the number of lanes"
        super().__init__()
        self.width, self.size, self.length, self.lanes = width, size, length, lanes
        outer_edge = size + length
        segs: Dict[str, AbstractLane] = {}
        for i in range(lanes):
            off = width * (i + 1/2)
            # Outer straights
            segs[f"NR{i}"] = StraightLane(f"NR{i}", (-off, outer_edge), (-off, size), width)
            segs[f"SR{i}"] = StraightLane(f"SR{i}", (off, -outer_edge), (off, -size), width)
            segs[f"ER{i}"] = StraightLane(f"ER{i}", (outer_edge, off), (size, off), width)
            segs[f"WR{i}"] = StraightLane(f"WR{i}", (-outer_edge, -off), (-size, -off), width)
            segs[f"NL{i}"] = StraightLane(f"NL{i}", (off, size), (off, outer_edge), width)
            segs[f"EL{i}"] = StraightLane(f"EL{i}", (size, -off), (outer_edge, -off), width)
            segs[f"SL{i}"] = StraightLane(f"SL{i}", (-off, -size), (-off, -outer_edge), width)
            segs[f"WL{i}"] = StraightLane(f"WL{i}", (-size, off), (-outer_edge, off), width)
            # Inner straights
            segs[f"NSR{i}"] = StraightLane(f"NSR{i}", (-off, size), (-off, -size), width)
            segs[f"NSL{i}"] = StraightLane(f"NSL{i}", (off, -size), (off, size), width)
            segs[f"WER{i}"] = StraightLane(f"WER{i}", (-size, -off), (size, -off), width)
            segs[f"WEL{i}"] = StraightLane(f"WEL{i}", (size, off), (-size, off), width)
        curve_params = {
            "NW": ((-size, size), 3),
            "NE": ((size, size), 2),
            "SW": ((-size, -size), 0),
            "SE": ((size, -size), 1),
        }
        segs.update(
            {
                (n := k + "IO"[io] + str(lane)): CircularLane(
                    n,
                    center,
                    size + width * (lane + 1/2) * [-1, 1][io],
                    (start + (int(io == 0))) * math.pi / 2,     # XXX inner circulars are clockwise, thus start larger then end
                    (start + (int(io == 1))) * math.pi / 2,     # also python bool converts to int like C would
                    clockwise=io == 0,
                    width=width,
                )
                for k, (center, start) in curve_params.items()
                for io in range(2) for lane in range(lanes)
            }
        )
        # straight
        straights = [
            ("NS", "NR", "NSR", "SL"),
            ("SN", "SR", "NSL", "NL"),
            ("WE", "WR", "WER", "EL"),
            ("EW", "ER", "WEL", "WL"),
        ]
        self.add_lanes([Lane(f"{straight[0]}.{i}", [segs[seg + str(i)] for seg in straight[1:]]) for straight in straights for i in range(lanes)])
        # curve
        curves = ["NWI", "NEO", "WNO", "ENI", "SWO", "SEI", "WSI", "ESO"]
        self.add_lanes([Lane(f"{s}{e}.{i}", [segs[s + "R" + str(i)], segs.get(s + e + d + str(i), None) or segs[e + s + d + str(i)], segs[e + "L" + str(i)]])
                        for s, e, d in curves for i in range(lanes)])

    def h(self, lane_idx: str, mode_src: str, mode_dest: str) -> Optional[str]:
        # ret = self._h(lane_idx, mode_src, mode_dest)
        # print("H", lane_idx, mode_src, mode_dest, "->", ret)
        # return ret

    # def _h(self, lane_idx: str, mode_src: str, mode_dest: str) -> Optional[str]:
        src_sw, dst_sw = mode_src.startswith("Switch"), mode_dest.startswith("Switch")
        if src_sw:
            if dst_sw:
                return None
            else:
                lanes, ind = lane_idx.split(".")
                ind = int(ind)
                if "Left" in mode_dest and ind > 0:
                    return f"{lanes}.{ind - 1}"
                if "Right" in mode_dest and ind < self.lanes - 1:
                    return f"{lanes}.{ind + 1}"
                return None
        else:
            if dst_sw:
                return lane_idx
            else:
                return lane_idx
