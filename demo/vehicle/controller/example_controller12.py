from enum import Enum, auto
import copy
from typing import List

class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()        # Pedestrians
    Sign = auto()       # Signs, stop signs, merge, yield etc.
    Signal = auto()     # Traffic lights
    Obstacle = auto()   # Static (to road/lane) obstacles

class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()
    Stop = auto()

class LaneMode(Enum):
    Lane0 = auto()
    Lane1 = auto()
    Lane2 = auto()

class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    vehicle_mode: VehicleMode = VehicleMode.Normal
    lane_mode: LaneMode = LaneMode.Lane0
    type_mode: LaneObjectMode = LaneObjectMode.Vehicle

    def __init__(self, x, y, theta, v, vehicle_mode: VehicleMode, lane_mode: LaneMode, type_mode: LaneObjectMode):
        pass

def controller(ego:State, others:List[State], lane_map):
    output = copy.deepcopy(ego)
    # Detect the stop sign
    if ego.vehicle_mode == VehicleMode.Normal:
        if any(other.x - ego.x < 5 and other.x - ego.x > -1 for other in others):
            output.vehicle_mode = VehicleMode.Brake 
    if ego.vehicle_mode == VehicleMode.Brake:
        if ego.v <= 0:
            output.vehicle_mode = VehicleMode.Stop
            output.v = 0
    return output

