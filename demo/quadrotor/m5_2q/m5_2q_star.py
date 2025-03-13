from quadrotor_agent import QuadrotorAgent
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import reachtube_tree
from verse.plotter.plotter3D_new import *
from verse.plotter.plotter3D import *
from verse.map.example_map.map_tacas import M5
import pyvista as pv
from enum import Enum, auto
import warnings
import sys
from verse.analysis.analysis_tree import AnalysisTreeNodeType
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *
import time
from verse.plotter.plotterStar import *
from verse.utils.star_diams import *

warnings.filterwarnings("ignore")


class CraftMode(Enum):
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
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))

    input_code_name = os.path.join(script_dir, "quadrotor_controller3.py")
    input_code_name2 = os.path.join(script_dir, "quadrotor_controller4.py")

    # bench = Benchmark(sys.argv)
    scenario = Scenario(ScenarioConfig(parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    time_step = 0.2
    quadrotor1 = QuadrotorAgent(
        "test1", file_name=input_code_name, t_v_pair=(1, 1), box_side=[0.4] * 3
    )
    init_l_1 = [1.5, -0.5, -0.5, 0, 0, 0]
    init_u_1 = [2.5, 0.5, 0.5, 0, 0, 0]
    init_1 = from_rect(init_l_1, init_u_1)
    # print(init_1, init_1.overapprox_rectangle())
    # exit()
    quadrotor1.set_initial(init_1, (CraftMode.Normal, TrackMode.T1))
    scenario.add_agent(quadrotor1)

    quadrotor2 = QuadrotorAgent(
        "test2", file_name=input_code_name2, t_v_pair=(1, 0.5), box_side=[0.4] * 3
    )
    init_l_2 = [19.5, -0.5, -0.5, 0, 0, 0]
    init_u_2 = [20.5, 0.5, 0.5, 0, 0, 0]
    
    init_2 = from_rect(init_l_2, init_u_2)
    # print(init_2, init_2.overapprox_rectangle())
    # exit()
    quadrotor2.set_initial(init_2, (CraftMode.Normal, TrackMode.T1))
    scenario.add_agent(quadrotor2)

    tmp_map = M5()
    scenario.set_map(tmp_map)
    scenario.set_sensor(BaseStarSensor())
    # scenario.set_sensor(QuadrotorSensor())

    # traces = scenario.simulate(40, time_step, seed=4)
    # fig = go.Figure()
    # fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [1, 2, 3])
    # fig.show()

    # if bench.config.compare:
    #     traces1, traces2 = bench.compare_run(60, time_step)
    #     exit(0)
    # traces = bench.run(60, time_step)
    start = time.perf_counter()
    traces = scenario.verify(60, time_step)
    # if bench.config.dump:
        # traces.dump(os.path.join(script_dir, "output9_dryvr.json"))
    print(f'Verification time: {time.perf_counter()-start}')
    diams = time_step_diameter(traces, 60, time_step)
    print(f'Initial diameter: {diams[0]}\n Final: {diams[-1]}\n Average: {sum(diams)/len(diams)}')

    # traces = scenario.verify(60, time_step,
    #                          reachability_method='NeuReach',
    #                          params={
    #                              "N_X0": 1,
    #                              "N_x0": 50,
    #                              "N_t": 50,
    #                              "epochs": 50,
    #                              "_lambda": 10,
    #                              "use_cuda": True,
    #                              'r': 0,
    #                          }
    #                         )
    # traces.dump('./demo/tacas2023/exp9/output9_NeuReach.json')
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 1, [0,1])
    # fig.show()
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 2, [0,1])
    # fig.show()
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 3, [0,1])
    # fig.show()
    # traces = AnalysisTree.load('./demo/tacas2023/exp9/output9.json')
    # if len(sys.argv)>1 and sys.argv[1]=='pl':

    # if bench.config.plot:
    #     if traces.type == AnalysisTreeNodeType.REACH_TUBE:
    #         fig = pv.Plotter()
    #         fig = plot3dMap(tmp_map, ax=fig)
    #         fig = plot3dReachtube(traces, "test1", 1, 2, 3, color="r", ax=fig)
    #         fig = plot3dReachtube(traces, "test2", 1, 2, 3, color="b", ax=fig)
    #         fig = plot_line_3d([0, 0, 0], [10, 0, 0], fig, "r", line_width=5)
    #         fig = plot_line_3d([0, 0, 0], [0, 10, 0], fig, "g", line_width=5)
    #         fig = plot_line_3d([0, 0, 0], [0, 0, 10], fig, "b", line_width=5)
    #         fig.set_background("#e0e0e0")
    #         fig.show()
    #     else:
    #         fig = go.Figure()
    #         fig = draw_map_3d(tmp_map, fig, fill_type="center")
    #         fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [1, 2, 3])
    #         fig.update_layout(
    #             scene=dict(
    #                 # xaxis = dict(nticks=4, range=[],),
    #                 yaxis = dict(nticks=4, range=[-10,10],),
    #                 zaxis=dict(
    #                     nticks=4,
    #                     range=[-12, 12],
    #                 )
    #             )
    #         )
    #         fig.show()  
    # # elif len(sys.argv)>1 and sys.argv[1]=='pc':
    # #     fig = go.Figure()
    # #     fig = reachtube_tree(traces, None, fig, 0, 1, [0,1])
    # #     fig.show()
    # #     fig = go.Figure()
    # #     fig = reachtube_tree(traces, None, fig, 0, 3, [0,1])
    # #     fig.show()
    # bench.report()
    # fig = go.Figure()
    # fig = reachtube_tree_3d(
    #     traces,
    #     tmp_map,
    #     fig, 1, 2, 3,
    #     [1, 2, 3],
    #     map_type='center',
    # )
    # fig.show()
