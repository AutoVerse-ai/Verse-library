from wsgiref.validate import PartialIteratorWrapper
from mp0 import VehicleAgent, PedestrainAgent, VehiclePedestrainSensor, verify_refine
from verse import Scenario, ScenarioConfig

from verse.plotter.plotter2D import *

import plotly.graph_objects as go
import copy

if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "vehicle_controller.py")
    vehicle = VehicleAgent('car', file_name=input_code_name)
    input_code_name = os.path.join(script_dir, "pedestrain_controller.py")
    pedestrain = PedestrainAgent('pedestrain')

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(vehicle) 
    scenario.add_agent(pedestrain)
    scenario.set_sensor(VehiclePedestrainSensor())

    trace_list = verify_refine([[-5,0,0,5,0],[5,0,0,10.0,0]], [[80,-30,0,3,0],[80,-30,0,3,0]], scenario)
    fig = go.Figure()
    for traces in trace_list:
        fig = reachtube_tree(traces, None, fig, 0,1,[0,1],'lines', 'trace')
    fig.show()
