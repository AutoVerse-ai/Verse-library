from enum import Enum,auto

class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()

class LaneMode(Enum):
    Lane0 = auto()

def controller(x,y,theta,v,vehicle_mode, lane_mode):
    output_vehicle_mode = vehicle_mode
    output_lane_mode = lane_mode
    if vehicle_mode == VehicleMode.Normal:
        if lane_mode == LaneMode.Lane0:
            if x > 3 and x < 5:
                output_vehicle_mode = VehicleMode.SwitchLeft
            if x > 3 and x < 5:
                output_vehicle_mode = VehicleMode.SwitchRight
    if vehicle_mode == VehicleMode.SwitchLeft:
        if lane_mode == LaneMode.Lane0:
            if x > 10:
                output_vehicle_mode = VehicleMode.Normal
    if vehicle_mode == VehicleMode.SwitchRight:
        if lane_mode == LaneMode.Lane0:
            if x > 10:
                output_vehicle_mode = VehicleMode.Normal

    return output_vehicle_mode, output_lane_mode
        
from ourtool.agents.car_agent import CarAgent
from ourtool.scenario.scenario import Scenario
import matplotlib.pyplot as plt 
import numpy as np

if __name__ == "__main__":
    input_code_name = 'car_lane_switch.py'
    scenario = Scenario()
    
    car = CarAgent('ego', file_name=input_code_name)
    scenario.add_agent(car)
    
    # simulator = Simulator()
    traces = scenario.simulate(
        [[0,0,0,0.5]],
        [(VehicleMode.Normal, LaneMode.Lane0)],
        [car],
        scenario,
        40
    )

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
