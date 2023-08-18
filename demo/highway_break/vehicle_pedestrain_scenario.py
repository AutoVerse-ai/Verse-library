from wsgiref.validate import PartialIteratorWrapper
from vehicle_agent import VehicleAgent
from pedestrain_agent import PedestrainAgent
from vehicle_pedestrain_sensor import VehiclePedestrainSensor
from verse import Scenario, ScenarioConfig

from vehicle_controller import VehicleMode
from pedestrain_controller import PedestrainMode 
from vehicle_controller import State
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
import copy

def tree_safe(tree: AnalysisTree):
    for node in tree.nodes:
        if node.assert_hits is not None:
            return False 
    return True

def verify_refine(init_car, init_ped, scenario: Scenario):
    partition_depth = 0
    init_queue = [(init_car, init_ped, partition_depth)]
    res_list = []
    while init_queue!=[] and partition_depth < 10:
        car_init, ped_init, partition_depth = init_queue.pop(0)
        print(f"######## {partition_depth}, {car_init[0][3]}, {car_init[1][3]}")
        scenario.set_init_single('car', car_init, (VehicleMode.Normal,))
        scenario.set_init_single('pedestrain', ped_init, (PedestrainMode.Normal,))
        traces = scenario.verify(30, 0.05)
        if not tree_safe(traces):
            # Partition car and pedestrain initial state
            # if partition_depth%3==0:
            #     car_x_init = (car_init[0][0] + car_init[1][0])/2
            #     car_init1 = copy.deepcopy(car_init)
            #     car_init1[1][0] = car_x_init 
            #     init_queue.append((car_init1, ped_init, partition_depth+1))
            #     car_init2 = copy.deepcopy(car_init)
            #     car_init2[0][0] = car_x_init 
            #     init_queue.append((car_init2, ped_init, partition_depth+1))
            # else:
            if car_init[1][3] - car_init[0][3] < 0.01 or partition_depth >= 10:
                print('Threshold Reached. Stop Refining')
                res_list.append(traces)
                continue
            car_v_init = (car_init[0][3] + car_init[1][3])/2
            car_init1 = copy.deepcopy(car_init)
            car_init1[1][3] = car_v_init 
            init_queue.append((car_init1, ped_init, partition_depth+1))
            car_init2 = copy.deepcopy(car_init)
            car_init2[0][3] = car_v_init 
            init_queue.append((car_init2, ped_init, partition_depth+1))
        else:
            res_list.append(traces)
    return res_list

if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "vehicle_controller.py")
    vehicle = VehicleAgent('car', file_name=input_code_name)
    input_code_name = os.path.join(script_dir, "pedestrain_controller.py")
    pedestrain = PedestrainAgent('pedestrain', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(vehicle) 
    scenario.add_agent(pedestrain)
    scenario.set_sensor(VehiclePedestrainSensor())

    # scenario.set_init_single(
    #     'car', [[0,0,0,9.0,0],[2,0,0,10.0,0]],(VehicleMode.Normal,)
    # )
    # scenario.set_init_single(
    #     'pedestrain', [[80,-30,0,3,0],[80,-30,0,3,0]], (PedestrainMode.Normal,)
    # )


    # traces = scenario.verify(30, 0.05)
    trace_list = verify_refine([[0,0,0,5,0],[2,0,0,10.0,0]], [[80,-30,0,3,0],[80,-30,0,3,0]], scenario)
    # fig = go.Figure()
    # fig = reachtube_anime(traces, None, fig, 1, 2)
    # fig.show()
    fig = go.Figure()
    for traces in trace_list:
        fig = reachtube_tree(traces, None, fig, 0,1,[0,1],'lines', 'trace')
    fig.show()
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0,2,[0,1],'lines', 'trace')
    # fig.show()

    # Simulate multi
    # Plotter
    # Manual refinement
    # Compute average speed to simulation
    # Monday -> two graphs
