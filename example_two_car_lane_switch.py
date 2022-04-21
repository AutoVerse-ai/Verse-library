from enum import Enum,auto

class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()

class LaneMode(Enum):
    Lane0 = auto()
    Lane1 = auto()

class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    vehicle_mode:VehicleMode = VehicleMode.Normal
    lane_mode:LaneMode = LaneMode.Lane0

    def __init__(self):
        self.data = []

def controller(ego:State, other:State, map):
    output = ego
    if ego.vehicle_mode == VehicleMode.Normal:
        if ego.lane_mode == LaneMode.Lane0:
            if other.x - ego.x > 3 and other.x - ego.x < 5 and map.has_left(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchLeft
                output.lane_mode = map.left_lane(ego.lane_mode)
            if other.x - ego.x > 3 and other.x - ego.x < 5:
                output.vehicle_mode = VehicleMode.SwitchRight
    if ego.vehicle_mode == VehicleMode.SwitchLeft:
        if ego.lane_mode == LaneMode.Lane0:
            if ego.x - other.x > 10:
                output.vehicle_mode = VehicleMode.Normal
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if ego.lane_mode == LaneMode.Lane0:
            if ego.x - other.x > 10:
                output.vehicle_mode = VehicleMode.Normal

    return output
    
from ourtool.agents.car_agent import CarAgent
from ourtool.scenario.scenario import Scenario
from user.sensor import SimpleSensor
from user.map import SimpleMap
import matplotlib.pyplot as plt 
import numpy as np

if __name__ == "__main__":
    input_code_name = 'example_two_car_lane_switch.py'
    scenario = Scenario()
    
    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = CarAgent('car2', file_name=input_code_name)
    scenario.add_agent(car)
    scenario.add_map(SimpleMap())
    # scenario.set_sensor(SimpleSensor())
    scenario.set_init(
        [[0,0,0,1.0], [10,0,0,0.5]],
        [
            (VehicleMode.Normal, LaneMode.Lane0),
            (VehicleMode.Normal, LaneMode.Lane0)
        ]
    )
    # simulator = Simulator()
    traces = scenario.simulate(40)

    queue = [traces]
    while queue!=[]:
        node = queue.pop(0)
        traces = node.trace
        agent_id = 'ego'
        # for agent_id in traces:
        trace = np.array(traces[agent_id])
        plt.plot(trace[:,0], trace[:,2], 'b')
        # if node.child != []:
        queue += node.child 
    plt.show()
