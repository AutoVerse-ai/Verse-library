from mp0_p1 import VehicleAgent, PedestrianAgent, VehiclePedestrianSensor
from verse import Scenario, ScenarioConfig
from vehicle_controller import VehicleMode, PedestrianMode

from verse.plotter.plotter3D import *
from verse.plotter.plotter3D_new import *
import pyvista as pv
import plotly.graph_objects as go
import copy

if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "vehicle_controller.py")
    vehicle = VehicleAgent('car', file_name=input_code_name)
    pedestrian = PedestrianAgent('pedestrian')

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(vehicle) 
    scenario.add_agent(pedestrian)
    scenario.set_sensor(VehiclePedestrianSensor())


    # init_car = [[4,-5,0,8],[5,5,0,8]]
    # init_pedestrian = [[170,-55,0,3],[175,-52,0,5]]

    init_car = [[-5, -5 ,0,8],[5,5,0,8]]
    init_pedestrian = [[170, -55, 0,3],[175, -53,0,5]]

    scenario.set_init_single(
        'car', init_car,(VehicleMode.Normal,)
    )
    scenario.set_init_single(
        'pedestrian', init_pedestrian, (PedestrianMode.Normal,)
    )

            # traces = []
            # fig = go.Figure()
            # n=3
            # for i in range(n):
            #     trace = scenario.simulate(50, 0.1)
            #     traces.append(trace)
            #     fig = simulation_tree_3d(trace, fig,\
            #                             0,'time', 1,'x',2,'y')
            # avg_vel, unsafe_frac, unsafe_init = eval_velocity(traces)
            # fig.show()
            #fig = pv.Plotter()
            #fig.show(interactive_update=True)

    # # ----------- verify no refine: Uncomment this block to perform verification without refinement ----------

    fig = pv.Plotter()
    fig.show(interactive_update=True)
    traces = scenario.verify(80, 0.1,fig )

    fig = plot3dReachtube(traces,'car',0,1,2,'b',fig)
    fig = plot3dReachtube(traces,'pedestrian',0,1,2,'r',fig)
    fig.show()
    
    # fig = go.Figure()
    # fig = reachtube_tree_3d(traces, fig,\
    #                           0,'time', 1,'x',2,'y')
    # fig.show()
    # # -----------------------------------------
