from wsgiref.validate import PartialIteratorWrapper
from mp0 import VehicleAgent, PedestrainAgent, VehiclePedestrainSensor, verify_refine, eval_velocity, sample_init
from verse import Scenario, ScenarioConfig
from vehicle_controller import VehicleMode, PedestrainMode

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
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

    scenario.set_init_single(
        'car', [[0,0,0,9.0,0],[2,0,0,10.0,0]],(VehicleMode.Normal,)
    )
    scenario.set_init_single(
        'pedestrain', [[80,-30,0,3,0],[80,-30,0,3,0]], (PedestrainMode.Normal,)
    )

    # # ----------- Simulate multi -------------
    init_dict_list= sample_init(scenario.init_dict)
    trace_list = scenario.simulate_multi(30, 0.5, init_dict_list=init_dict_list)
    fig = go.Figure()
    for trace in trace_list:
        fig = simulation_tree_3d(trace, None, fig, 0,'time', 1,'x',2,'y',[0,1,2])
    fig.show()
    eval_velocity(trace_list, 'car')
    # # -----------------------------------------

    # ----------- verify old version ----------
    # traces = scenario.verify(30, 1)
    # # fig = go.Figure()
    # # fig = reachtube_tree(traces, None, fig, 0,1,[0,1],'lines', 'trace')
    # # fig.show()
    # # fig = go.Figure()
    # # fig = reachtube_tree(traces, None, fig, 0,2,[0,1],'lines', 'trace')
    # # fig.show()

    # fig = go.Figure()
    # fig = reachtube_tree_3d(traces, None, fig, 0,'time', 1,'x',2,'y',[0,1,2])
    # fig.show()

    # -----------------------------------------

    # ------------- Verify refine -------------
    com_traces = verify_refine([[0,0,0,5,0],[2,0,0,10.0,0]], [[80,-30,0,3,0],[80,-30,0,3,0]], scenario)
    # fig = go.Figure()
    # fig = reachtube_anime(traces, None, fig, 1, 2)
    # fig.show()
    # fig = go.Figure()
    # for traces in trace_list:
    #     fig = go.Figure()
    #     # fig = reachtube_tree(traces, None, fig, 0,1,[0,1],'lines', 'trace')
    #     # fig = reachtube_tree(traces, None, fig, 0,2,[0,1],'lines', 'trace')
    #     fig = reachtube_tree_3d(traces, None, fig, 0,'time', 1,'x',2,'y',[0,1,2])
    #     fig.show()

    fig = go.Figure()
    fig = reachtube_tree_3d(com_traces, None, fig, 0,'time', 1,'x',2,'y',[0,1,2])
    fig.show()
    # -----------------------------------------
