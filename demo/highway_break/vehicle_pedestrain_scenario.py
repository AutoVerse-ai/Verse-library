from vehicle_agent import VehicleAgent
from pedestrain_agent import PedestrainAgent
from vehicle_pedestrain_sensor import VehiclePedestrainSensor
from verse import Scenario, ScenarioConfig

from vehicle_controller import VehicleMode
from pedestrain_controller import PedestrainMode 
from vehicle_controller import State
from verse.plotter.plotter2D import *

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
        'car', [[0,0,0,10,0],[2,0,0,10,0]],(VehicleMode.Normal,)
    )
    scenario.set_init_single(
        'pedestrain', [[80,-30,0,3,0],[80,-30,0,3,0]], (PedestrainMode.Normal,)
    )

    scenario.set_sensor(VehiclePedestrainSensor())

    traces = scenario.verify(30, 0.05)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 0,1,[0,1],'lines', 'trace')
    fig.show()
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 0,2,[0,1],'lines', 'trace')
    fig.show()