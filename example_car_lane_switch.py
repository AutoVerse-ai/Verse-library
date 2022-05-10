from enum import Enum,auto

class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()

class LaneMode(Enum):
    Lane0 = auto()

class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    vehicle_mode: VehicleMode = VehicleMode.Normal
    lane_mode: LaneMode = LaneMode.Lane0

    def __init__(self, x, y, theta, v, vehicle_mode: VehicleMode, lane_mode: LaneMode):
        self.data = []

def controller(ego:State):
    output_vehicle_mode = ego.vehicle_mode
    output_lane_mode = ego.lane_mode
    if ego.vehicle_mode == VehicleMode.Normal:
        if ego.lane_mode == LaneMode.Lane0:
            if ego.x > 3 and ego.x < 5:
                output_vehicle_mode = VehicleMode.SwitchLeft
            if ego.x > 3 and ego.x < 5:
                output_vehicle_mode = VehicleMode.SwitchRight
    if ego.vehicle_mode == VehicleMode.SwitchLeft:
        if ego.lane_mode == LaneMode.Lane0:
            if ego.x > 10:
                output_vehicle_mode = VehicleMode.Normal
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if ego.lane_mode == LaneMode.Lane0:
            if ego.x > 10:
                output_vehicle_mode = VehicleMode.Normal

    return output_vehicle_mode, output_lane_mode
        
from src.example.example_agent.car_agent import CarAgent
from src.scene_verifier.scenario.scenario import Scenario
from src.example.example_map.simple_map import SimpleMap2
from src.example.example_sensor.fake_sensor import FakeSensor1
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    input_code_name = 'example_car_lane_switch.py'
    scenario = Scenario()
    
    car = CarAgent('ego', file_name=input_code_name)
    scenario.add_agent(car)
    scenario.add_map(SimpleMap2())
    scenario.set_sensor(FakeSensor1())
    
    # simulator = Simulator()
    scenario.set_init(
        [[0,3,0,0.5]],
        [(VehicleMode.Normal, LaneMode.Lane0)]
    )

    traces = scenario.simulate(
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
