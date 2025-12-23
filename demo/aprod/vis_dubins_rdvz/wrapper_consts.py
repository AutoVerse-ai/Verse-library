from typing import List, Dict, Callable
from bounded_angular import angular_span_rect_parser, angular_bounds_diff_correct
from bounded_map import get_heading_bounds_optimized, get_lateral_distance_bounds_optimized

ALIASES = {
    "atan2": "angular_span_rect_parser",
    "arctan2": "angular_span_rect_parser",
    "get_lane_heading": "get_heading_bounds_optimized",
    "get_lateral_distance": "get_lateral_distance_bounds_optimized",
    "minus_angular": "angular_bounds_diff_correct"
}

ANGULAR_FUNCTIONS = [
    "angular_span_rect_parser", "angular_bounds_diff_correct"
]

MAP_FUNCTIONS: List[str] = [
    # NOTE: in parser_wrapper, assume that lane_idx and lane_map are constants passed in -- the first two arguments of these two functions should just be given
    "get_heading_bounds_optimized",
    "get_lateral_distance_bounds_optimized",
]

# NOTE: don't know if I need this, but it could be useful if I don't want to keep using the 
FUNC_DICT: Dict[str, Callable] = {
    "angular_span_rect_parser": angular_span_rect_parser,
    "get_heading_bounds_optimized": get_heading_bounds_optimized,
    "get_lateral_distance_bounds_optimized": get_lateral_distance_bounds_optimized,
    "angular_bounds_diff_correct": angular_bounds_diff_correct,
}