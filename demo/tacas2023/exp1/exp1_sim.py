from quadrotor_agent import QuadrotorAgent
from verse import Scenario
from verse.plotter.plotter3D_new import *
from verse.plotter.plotter3D import *
from verse.map.example_map.map_tacas import M6 
import pyvista as pv
from enum import Enum, auto
import warnings
import time
import sys
warnings.filterwarnings("ignore")

class TacticalMode(Enum):
    Normal = auto()
    MoveUp = auto()
    MoveDown = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M10 = auto()
    M12 = auto()
    M21 = auto()


if __name__ == "__main__":
    input_code_name = './demo/tacas2023/exp1/quadrotor_controller3.py'

    scenario = Scenario()
    time_step = 0.1
    quadrotor1 = QuadrotorAgent(
        'test1', file_name=input_code_name, t_v_pair=(1, 1), box_side=[0.4]*3)
    init_l_1 = [9.5, 0, -0.35, 0, 0, 0]
    init_u_1 = [10.2, 0.7, 0.35, 0, 0, 0]
    quadrotor1.set_initial(
        [init_l_1, init_u_1],
        (TacticalMode.Normal, TrackMode.T1)
    )
    scenario.add_agent(quadrotor1)

    quadrotor2 = QuadrotorAgent(
        'test2', file_name=input_code_name, t_v_pair=(1, 0.3), box_side=[0.4]*3)
    init_l_2 = [3, 9, -0.35, 0, 0, 0]
    init_u_2 = [3.7, 9.7, 0.35, 0, 0, 0]
    quadrotor2.set_initial(
        [init_l_2, init_u_2],
        (TacticalMode.Normal, TrackMode.T1)
    )
    scenario.add_agent(quadrotor2)

    tmp_map = M6()
    scenario.set_map(tmp_map)
    # scenario.set_sensor(QuadrotorSensor())

    # traces = scenario.simulate(40, time_step, seed=4)
    # traces.dump("./output1.json")
    # traces = AnalysisTree.load('./output1.json')
    start_time = time.time()
    traces = scenario.simulate_simple(40, time_step, seed=4)
    run_time = time.time() - start_time
    traces.dump('demo/tacas2023/exp1/output1_sim.json')
    # print({
    #     "#A": len(scenario.agent_dict),
    #     "A": "Q",
    #     "Map": "M6",
    #     "postCont": "DryVR",
    #     "Noisy S": "No",
    #     "# Tr": len(traces.nodes),
    #     "Run Time": run_time,
    # })
    if len(sys.argv) > 1 and sys.argv[1]=='p':
        fig = go.Figure()
        fig = draw_map_3d(tmp_map, fig, fill_type = 'center')
        fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [1, 2, 3])
        fig.update_layout(
            scene=dict(
                # xaxis = dict(nticks=4, range=xrange,),
                # yaxis = dict(nticks=4, range=yrange,),
                zaxis = dict(nticks=4, range=[-12,12],)
            )
        )
        fig.show()


