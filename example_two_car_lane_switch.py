from enum import Enum,auto

from ourtool.map.lane_map import LaneMap

class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()

class LaneMode(Enum):
    Lane0 = auto()
    Lane1 = auto()
    Lane2 = auto()

class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    vehicle_mode:VehicleMode = VehicleMode.Normal
    lane_mode:LaneMode = LaneMode.Lane0

    def __init__(self,x,y,theta,v,vehicle_mode:VehicleMode, lane_mode:LaneMode):
        self.data = []

def controller(ego:State, other:State, lane_map):
    output = ego
    if ego.vehicle_mode == VehicleMode.Normal:
        if other.x - ego.x > 3 and other.x - ego.x < 5 and ego.lane_mode == other.lane_mode:
            if lane_map.has_left(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchLeft
        if other.x - ego.x > 3 and other.x - ego.x < 5 and ego.lane_mode == other.lane_mode:
            if lane_map.has_right(ego.lane_mode):
                output.vehicle_mode = VehicleMode.SwitchRight
    if ego.vehicle_mode == VehicleMode.SwitchLeft:
        if ego.y >= lane_map.lane_geometry(ego.lane_mode)-0.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.left_lane(ego.lane_mode)
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if ego.y <= lane_map.lane_geometry(ego.lane_mode)+0.5:
            output.vehicle_mode = VehicleMode.Normal
            output.lane_mode = lane_map.right_lane(ego.lane_mode)
    
    return output
    
from ourtool.agents.car_agent import CarAgent
from ourtool.scenario.scenario import Scenario
# from user.simple_sensor import SimpleSensor
from user.simple_map import SimpleMap, SimpleMap2
import matplotlib.pyplot as plt 
import numpy as np

if __name__ == "__main__":
    input_code_name = 'example_two_car_lane_switch.py'
    scenario = Scenario()
    
    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = CarAgent('car2', file_name=input_code_name)
    scenario.add_agent(car)
    scenario.add_map(SimpleMap2())
    # scenario.set_sensor(SimpleSensor())
    scenario.set_init(
        [[10,0,0,0.5], [0,-3,0,1.0]],
        [
            (VehicleMode.Normal, LaneMode.Lane1),
            (VehicleMode.Normal, LaneMode.Lane2)
        ]
    )
    # simulator = Simulator()
    traces = scenario.simulate(40)

    plt.plot([0,40],[3,3],'g')
    plt.plot([0,40],[0,0],'g')
    plt.plot([0,40],[-3,-3],'g')

    queue = [traces]
    while queue!=[]:
        node = queue.pop(0)
        traces = node.trace
        # for agent_id in traces:
        agent_id = 'car2'
        trace = np.array(traces[agent_id])
        plt.plot(trace[:,1], trace[:,2], 'r')

        agent_id = 'car1'
        trace = np.array(traces[agent_id])
        plt.plot(trace[:,1], trace[:,2], 'b')

        # if node.child != []:
        queue += node.child 
    plt.show()
