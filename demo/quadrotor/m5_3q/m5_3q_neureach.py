from quadrotor_agent import QuadrotorAgent
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import reachtube_tree
from verse.plotter.plotter3D_new import *
from verse.plotter.plotter3D import *
from verse.map.example_map.simple_map_3d import SimpleMap7
import pyvista as pv
from enum import Enum, auto
import polytope as pc
import warnings
import sys

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
    input_code_name = os.path.join(script_dir, "./demo/tacas2023/exp10/quadrotor_controller3.py")
    input_code_name2 = os.path.join(script_dir, "./demo/tacas2023/exp10/quadrotor_controller4.py")

    bench = Benchmark(sys.argv, reachability_method="NeuReach")
    bench.agent_type = "D"
    bench.noisy_s = "No"
    time_step = 0.2
    quadrotor1 = QuadrotorAgent(
        "test1", file_name=input_code_name, t_v_pair=(1, 1), box_side=[0.4] * 3
    )
    init_l_1 = [1.5, -0.5, -0.5, 0, 0, 0]
    init_u_1 = [2.5, 0.5, 0.5, 0, 0, 0]
    quadrotor1.set_initial([init_l_1, init_u_1], (CraftMode.Normal, TrackMode.T1))
    bench.scenario.add_agent(quadrotor1)

    quadrotor2 = QuadrotorAgent(
        "test2", file_name=input_code_name2, t_v_pair=(1, 0.5), box_side=[0.4] * 3
    )
    init_l_2 = [19.5, -0.5, -0.5, 0, 0, 0]
    init_u_2 = [20.5, 0.5, 0.5, 0, 0, 0]
    quadrotor2.set_initial([init_l_2, init_u_2], (CraftMode.Normal, TrackMode.T1))
    bench.scenario.add_agent(quadrotor2)

    quadrotor3 = QuadrotorAgent(
        "test3", file_name=input_code_name2, t_v_pair=(1, 0.5), box_side=[0.4] * 3
    )
    init_l_3 = [5 + 39.5, -0.5, 8 - 0.5, 0, 0, 0]
    init_u_3 = [5 + 40.5, 0.5, 8 + 0.5, 0, 0, 0]
    quadrotor3.set_initial([init_l_3, init_u_3], (CraftMode.Normal, TrackMode.T0))
    bench.scenario.add_agent(quadrotor3)

    tmp_map = SimpleMap7()
    bench.scenario.set_map(tmp_map)
    # scenario.set_sensor(QuadrotorSensor())

    # traces = scenario.simulate(40, time_step, seed=4)
    # fig = go.Figure()
    # fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [1, 2, 3])
    # fig.show()

    # traces = scenario.verify(90, time_step)
    # traces.dump('./demo/tacas2023/exp10/output10.json')
    if bench.config.compare:
        traces1, traces2 = bench.compare_run(
            90,
            time_step,
            params={
                "N_X0": 1,
                "N_x0": 50,
                "N_t": 50,
                "epochs": 50,
                "_lambda": 10,
                "use_cuda": True,
                "r": 0,
            },
        )
        exit(0)
    traces = bench.run(
        90,
        time_step,
        params={
            "N_X0": 1,
            "N_x0": 50,
            "N_t": 50,
            "epochs": 50,
            "_lambda": 10,
            "use_cuda": True,
            "r": 0,
        },
    )
    if bench.config.dump:
        traces.dump(os.path.join(script_dir, "./demo/tacas2023/exp10/output10_neureach.json"))
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 1, [0,1])
    # fig.show()
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 2, [0,1])
    # fig.show()
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 3, [0,1])
    # fig.show()

    # traces = AnalysisTree.load('demo/tacas2023/exp10/output10.json')
    if bench.config.plot:
        fig = plot3dMap(tmp_map, ax=fig)
        fig = plot3dReachtube(traces, "test1", 1, 2, 3, color="r", ax=fig)
        fig = plot3dReachtube(traces, "test2", 1, 2, 3, color="b", ax=fig)
        fig = plot3dReachtube(traces, "test3", 1, 2, 3, color="g", ax=fig)
        fig = plot_line_3d([0, 0, 0], [10, 0, 0], fig, "r", line_width=5)
        fig = plot_line_3d([0, 0, 0], [0, 10, 0], fig, "g", line_width=5)
        fig = plot_line_3d([0, 0, 0], [0, 0, 10], fig, "b", line_width=5)
        fig.set_background("#e0e0e0")
        box = np.array([[40, -5, -10], [50, 5, -6]]).T
        poly = pc.box2poly(box)
        fig = plot_polytope_3d(poly.A, poly.b, fig, trans=0.1, color="yellow")
        fig.show()
    bench.report()
    # fig = go.Figure()
    # fig = reachtube_tree_3d(
    #     traces,
    #     tmp_map,
    #     fig, 1, 2, 3,
    #     [1, 2, 3],
    #     map_type='center',
    #     xrange = [0,100],
    #     yrange = [0,100],
    #     zrange = [-10,10]
    # )
    # fig.show()
