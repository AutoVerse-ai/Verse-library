from quadrotor_agent import QuadrotorAgent
from verse.analysis.analysis_tree import first_transitions
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import reachtube_tree
from verse.plotter.plotter3D_new import *
from verse.plotter.plotter3D import *
from verse.map.example_map.map_tacas import M5
from enum import Enum, auto
import warnings
import time
import sys
from verse.utils.star_diams import *
import time
from verse.analysis.verifier import ReachabilityMethod
from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *
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


sim_length = 75
time_step = 0.4


def run(meas=False):
    if bench.config.sim:
        bench.scenario.simulator.cache_hits = (0, 0)
    else:
        bench.scenario.verifier.tube_cache_hits = (0, 0)
        bench.scenario.verifier.trans_cache_hits = (0, 0)
    if not meas and not bench.scenario.config.incremental:
        return
    start = time.perf_counter()

    traces = bench.run(sim_length, time_step)
    print(f'Runtime: {time.perf_counter()-start}')
    diams = time_step_diameter_rect(traces, 60, time_step)
    print(f'Initial diameter: {diams[0]}\n Final: {diams[-1]}\n Average: {sum(diams)/len(diams)}')

    if bench.config.dump:
        traces.dump("tree2.json" if meas else "tree1.json")

    if bench.config.plot:
        import pyvista as pv
        import polytope as pc

        fig = pv.Plotter()
        fig = plot3dMap(bench.scenario.map, ax=fig)
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

    # if meas:
    bench.report()
    print(f"agent transition times: {first_transitions(traces)}")


if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "quadrotor_controller3.py")
    input_code_name2 = os.path.join(script_dir, "quadrotor_controller4.py")

    scenario = Scenario(ScenarioConfig(parallel=False))
    scenario.config.reachability_method =  ReachabilityMethod.STAR_SETS
    quadrotor1 = QuadrotorAgent(
        "test1", file_name=input_code_name, t_v_pair=(1, 1), box_side=[0.4] * 3
    )
    init_l_1 = [1.5, -0.5, -0.5, 0, 0, 0]
    init_u_1 = [2.5, 0.5, 0.5, 0, 0, 0]
    init_1 = from_rect(init_l_1, init_u_1)
    quadrotor1.set_initial(init_1, (CraftMode.Normal, TrackMode.T1))
    scenario.add_agent(quadrotor1)

    quadrotor2 = QuadrotorAgent(
        "test2", file_name=input_code_name2, t_v_pair=(1, 0.5), box_side=[0.4] * 3
    )
    init_l_2 = [19.5, -0.5, -0.5, 0, 0, 0]
    init_u_2 = [20.5, 0.5, 0.5, 0, 0, 0]
    init_2 = from_rect(init_l_2, init_u_2)
    quadrotor2.set_initial(init_2, (CraftMode.Normal, TrackMode.T1))
    scenario.add_agent(quadrotor2)

    quadrotor3 = QuadrotorAgent(
        "test3", file_name=input_code_name2, t_v_pair=(1, 0.5), box_side=[0.4] * 3
    )
    init_l_3 = [5 + 39.5, -0.5, 8 - 0.5, 0, 0, 0]
    init_u_3 = [5 + 40.5, 0.5, 8 + 0.5, 0, 0, 0]
    init_3 = from_rect(init_l_3, init_u_3)
    quadrotor3.set_initial(init_3, (CraftMode.Normal, TrackMode.T0))
    scenario.add_agent(quadrotor3)

    tmp_map = M5()
    scenario.set_map(tmp_map)
    scenario.set_sensor(BaseStarSensor())

    start = time.perf_counter()
    traces = scenario.verify(75, time_step)
    # if bench.config.dump:
        # traces.dump(os.path.join(script_dir, "output9_dryvr.json"))
    print(f'Verification time: {time.perf_counter()-start}')
    diams = time_step_diameter(traces, 75, time_step)
    print(f'Initial diameter: {diams[0]}\n Final: {diams[-1]}\n Average: {sum(diams)/len(diams)}')
    plot_reachtube_stars(traces, tmp_map, filter=1)
    exit()
    # scenario.set_sensor(QuadrotorSensor())

    # traces = scenario.simulate(40, time_step, seed=4)
    # fig = go.Figure()
    # fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [1, 2, 3])
    # fig.show()

    if "b" in bench.config.args:
        run(True)
    elif "r" in bench.config.args:
        run()
        run(True)
    elif "n" in bench.config.args:
        run()
        init_l_2 = [19, -0.5, -0.5, 0, 0, 0]
        init_u_2 = [20, 0.5, 0.5, 0, 0, 0]
        bench.scenario.set_init_single(
            "test2", [init_l_2, init_u_2], (CraftMode.Normal, TrackMode.T1)
        )
        run(True)
    elif "1" in bench.config.args:
        run()
        bench.swap_dl("test1", input_code_name2)
        run(True)
    elif "2" in bench.config.args:
        run()
        bench.swap_dl("test2", input_code_name)
        run(True)
    bench.report()
