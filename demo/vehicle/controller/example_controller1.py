from enum import Enum, auto
import copy

class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()

class TrackMode(Enum):
    Lane0 = auto()
    Lane1 = auto()
    Lane2 = auto()

class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    vehicle_mode: VehicleMode = VehicleMode.Normal
    lane_mode: TrackMode = TrackMode.Lane0

    def __init__(self, x, y, theta, v, vehicle_mode: VehicleMode, lane_mode: TrackMode):
        self.data = []

def controller(ego:State, other:State, lane_map):
    output = copy.deepcopy(ego)
    if ego.vehicle_mode == VehicleMode.Normal:
        if lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) > 0 \
        and lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) < 3 \
        and ego.lane_mode == other.lane_mode:
            output.vehicle_mode = VehicleMode.Brake
    elif ego.vehicle_mode == VehicleMode.Brake:
        if lane_map.get_longitudinal_position(other.lane_mode, [other.x,other.y]) - lane_map.get_longitudinal_position(ego.lane_mode, [ego.x,ego.y]) > 10 \
            or ego.lane_mode != other.lane_mode:
            output.vehicle_mode = VehicleMode.Normal

    return output
    
