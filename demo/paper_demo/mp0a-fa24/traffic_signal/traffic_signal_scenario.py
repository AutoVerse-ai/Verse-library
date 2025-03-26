from mp0_p2 import VehicleAgent, TrafficSignalAgent, TrafficSensor, verify_refine, eval_velocity, sample_init
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

    # # ----------- Different initial ranges -------------
    # # Uncomment this block to use R1
    init_car = [[0,-5,0,5],[50,5,0,5]]
    init_trfficlight = [[300,0,0,0,0],[300,0,0,0,0]]
    # # -----------------------------------------

    # # Uncomment this block to use R2
    #init_car = [[0,-5,0,5],[100,5,0,5]]
    #init_trfficlight = [[300,0,0,0,0],[300,0,0,0,0]]
    # # -----------------------------------------

    # # Uncomment this block to use R3
    # init_car = [[0,-5,0,0],[100,5,0,10]]
    # init_trfficlight = [[300,0,0,0,0],[300,0,0,0,0]]
    # # -----------------------------------------

    scenario.set_init_single(
        'car', init_car,(VehicleMode.Normal,)
    )
    scenario.set_init_single(
        'tl', init_trfficlight, (TLMode.GREEN,)
    )

    # ----------- Simulate simple: Uncomment this block to perform single simple simulation -------------
    #trace = scenario.simulate_simple(80, 0.1)
    #avg_vel, unsafe_frac, unsafe_init = eval_velocity([trace])
    #fig = go.Figure()
    #fig = simulation_tree_3d(trace, fig,\
    #                          0,'time', 1,'x',2,'y')
    #fig.show()
    # -----------------------------------------

    # # ----------- verify no refine: Uncomment this block to perform verification without refinement ----------
    traces = scenario.verify(80, 0.1)
    fig = go.Figure()
    fig = reachtube_tree_3d(traces, fig,\
                              0,'time', 1,'x',2,'y')
    fig.write_html('traffic2.html', auto_open=True)
    # # -----------------------------------------
    
    # ------------- Verify refine: Uncomment this block to perform verification with refinement -------------
    #traces = verify_refine(scenario, 80, 0.1)
    #fig = go.Figure()
    #for trace in traces:
    #    fig = reachtube_tree_3d(trace, fig,\
    #                         0,'time', 1,'x',2,'y')
    #fig.show()
    # -----------------------------------------
