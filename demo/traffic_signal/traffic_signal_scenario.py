from mp0 import VehicleAgent, TrafficSignalAgent, TrafficSensor, verify_refine, eval_velocity, sample_init
from verse import Scenario, ScenarioConfig
from vehicle_controller import VehicleMode, TLMode

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go
import copy

if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "vehicle_controller.py")
    vehicle = VehicleAgent('car', file_name=input_code_name)
    input_code_name = os.path.join(script_dir, "traffic_controller.py")
    tl = TrafficSignalAgent('tl', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(vehicle) 
    scenario.add_agent(tl)
    scenario.set_sensor(TrafficSensor())

    # # # R1
    # init_car = [[0,-5,0,5],[50,5,0,5]]
    # init_pedestrian = [[300,0,0,0,0],[300,0,0,0,0]]

    # # R2
    # init_car = [[0,-5,0,5],[100,5,0,5]]
    # init_pedestrian = [[300,0,0,0,0],[300,0,0,0,0]]

    # # R3
    init_car = [[0,-5,0,0],[100,5,0,10]]
    init_pedestrian = [[300,0,0,0,0],[300,0,0,0,0]]
    # 78.125, 81.25, car v, 3.75, 5.0
    # init_car = [[78.125,-5,0,3.75],[81.25,5,0,5]]
    # init_pedestrian = [[300,0,0,0,0],[300,0,0,0,0]]
    
    scenario.set_init_single(
        'car', init_car,(VehicleMode.Normal,)
    )
    scenario.set_init_single(
        'tl', init_pedestrian, (TLMode.GREEN,)
    )

    # # ----------- Simulate single -------------
    fig = go.Figure()
    for i in range(1):
        trace = scenario.simulate_simple(80, 0.1)
        fig = simulation_tree(trace, None, fig, 0, 1)
        print("=========================================================")
    fig.show()


    # fig = go.Figure()
    # for i in range(10):
    #     trace = scenario.simulate(80, 0.1)
    # # fig = simulation_tree_3d(trace, fig,\
    # #                           0,'time', 1,'x',2,'y')
    # # fig.show()
    #     fig = simulation_tree(trace, None, fig, 0, 1)
    # fig.show()
    # # ----------- Simulate multi -------------
    # init_dict_list= sample_init(scenario, num_sample=50)
    # trace_list = scenario.simulate_multi(100, 0.1,\
    #      init_dict_list=init_dict_list)
    # fig = go.Figure()
    # for trace in trace_list:
    #     # fig = simulation_tree_3d(trace, fig,\
    #     #                           0,'time', 1,'x',2,'y')
    #     fig = simulation_tree(trace, None, fig, 0, 1)
    # fig.show()
    # avg_vel, unsafe_frac, unsafe_init = eval_velocity(trace_list)
    # print(f"Average velocity {avg_vel}, Unsafe fraction {unsafe_frac}, Unsafe init {unsafe_init}")
    # # -----------------------------------------

    # ----------- verify old version ----------
    # traces = scenario.verify(100, 0.1)
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0,1,[0,1],'lines', 'trace')
    # fig.show()
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0,2,[0,1],'lines', 'trace')
    # fig.show()

    # fig = go.Figure()
    # fig = reachtube_tree_3d(traces, fig, 0,'time', 1,'x',2,'y')
    # fig.show()

    # -----------------------------------------

    # # ------------- Verify refine -------------
    res_list = verify_refine(scenario, 80, 0.1)
    fig = go.Figure()
    for trace in res_list:
        fig = reachtube_tree(trace, None, fig, 0, 1, label_mode = True)
    fig.show()
    # # -----------------------------------------
