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

def controller(ego_x, ego_y, ego_theta, ego_v, ego_vehicle_mode, ego_lane_mode, others_x, others_y, others_theta, others_v, others_vehicle_mode, others_lane_mode, map):
    # output = ego
    output_vehicle_mode = ego_vehicle_mode
    output_lane_mode = ego_lane_mode
    if ego_vehicle_mode == VehicleMode.Normal:
        if ego_lane_mode == LaneMode.Lane0:
            if others_x - ego_x > 3 and others_x - ego_x < 5 and map.can_swtich_left(ego_lane_mode):
                output_vehicle_mode = VehicleMode.SwitchLeft
                output_lane_mode = map.switch_left(ego_lane_mode)
            if others_x - ego_x > 3 and others_x - ego_x < 5:
                output_vehicle_mode = VehicleMode.SwitchRight
    if ego_vehicle_mode == VehicleMode.SwitchLeft:
        if ego_lane_mode == LaneMode.Lane0:
            if ego_x - others_x > 10:
                output_vehicle_mode = VehicleMode.Normal
    if ego_vehicle_mode == VehicleMode.SwitchRight:
        if ego_lane_mode == LaneMode.Lane0:
            if ego_x - others_x > 10:
                output_vehicle_mode = VehicleMode.Normal

    return output_vehicle_mode, output_lane_mode
    
from ourtool.agents.car_agent import CarAgent
from ourtool.scenario.scenario import Scenario
import matplotlib.pyplot as plt 
import numpy as np

if __name__ == "__main__":
    input_code_name = 'example_two_car_lane_switch.py'
    scenario = Scenario()
    
    car = CarAgent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    car = CarAgent('car2', file_name=input_code_name)
    scenario.add_agent(car)
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
