from typing import List
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional, Union

from dryvr_plus_plus.scene_verifier.utils.utils import wrap_to_pi, Vector, get_class_path, class_from_path, to_serializable
from dryvr_plus_plus.scene_verifier.map.lane_segment import LineType, AbstractLane, StraightLane


class StraightLane_3d(StraightLane):

    """A lane going in 3d straight line."""

    def __init__(self,
                 id: str,
                 start: Vector,
                 end: Vector,
                 width: float = AbstractLane.DEFAULT_WIDTH,
                 line_types: Tuple[LineType, LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        super().__init__(id, start ,end , width,line_types,forbidden,speed_limit,priority)
