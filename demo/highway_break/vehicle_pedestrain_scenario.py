from vehicle_agent import VehicleAgent
from pedestrain_agent import PedestrainAgent
from vehicle_pedestrain_sensor import VehiclePedestrainSensor
from verse import Scenario, ScenarioConfig

from vehicle_controller import VehicleMode
from pedestrain_controller import PedestrainMode 
from vehicle_controller import State
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import helper

import plotly.graph_objects as go

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

    scenario.set_init_single(
        'car', [[0,0,0,10,0],[5,5,0,10,0]],(VehicleMode.Normal,)
    )
    scenario.set_init_single(
        'pedestrain', [[80,-30,0,3,0],[80,-30,0,3,0]], (PedestrainMode.Normal,)
    )

    scenario.set_sensor(VehiclePedestrainSensor())

    # traces = scenario.verify(30, 0.1)
    # # helper.combine_tree([traces, traces])
    # # fig = go.Figure()
    # # fig = reachtube_tree(traces, None, fig, 0,1,[0,1],'lines', 'trace')
    # # fig.show()
    # # fig = go.Figure()
    # # fig = reachtube_tree(traces, None, fig, 0,2,[0,1],'lines', 'trace')
    # # fig.show()

    # fig = go.Figure()
    # fig = reachtube_tree_3d(traces, None, fig, 0,1,2,[0,1,2])
    # fig.show()

    init_dict_list= helper.sample(scenario.init_dict)
    # print(len(init_dict_list))
    trace_list = scenario.simulate_multi(30, 0.5, init_dict_list=init_dict_list)
    # helper.sample(scenario.init_dict)
    fig = go.Figure()
    for trace in trace_list:
        fig = simulation_tree_3d(trace, None, fig, 0,'time', 1,'x',2,'y',[0,1,2])
    fig.show()