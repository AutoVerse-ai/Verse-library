from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M2
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import *
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *
from enum import Enum, auto
import sys
import plotly.graph_objects as go
import time 
from verse.utils.star_diams import *

class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    T3 = auto()
    T4 = auto()
    M01 = auto()
    M12 = auto()
    M23 = auto()
    M40 = auto()
    M04 = auto()
    M32 = auto()
    M21 = auto()
    M10 = auto()


if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "example_controller7.py")

    # input_code_name = "./demo/tacas2023/exp3/example_controller7.py"
    # bench = Benchmark(sys.argv)

    scenario = Scenario(ScenarioConfig(parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS

    scenario.config.model_path = 'm2_highway'

    # bench.agent_type = "C"
    # bench.noisy_s = "No"
    C, g = new_pred(4)
    
    center = np.array([0.005, 0, 0, 1])
    basis = np.diag([0.005, 0.1, 0, 0])
    car1 = CarAgent("car1", file_name=input_code_name)
    car1.set_initial(
        initial_state= StarSet(center, basis, C, g),
        initial_mode=(AgentMode.Normal, TrackMode.T1)
    )
    scenario.add_agent(car1)

    car2 = NPCAgent("car2")
    center = np.array([10, 0, 0, 0.5])
    basis = np.diag([0, 0.1, 0, 0])
    car2.set_initial(
        initial_state= StarSet(center, basis, C, g),
        initial_mode=(AgentMode.Normal, TrackMode.T1)
    )
    scenario.add_agent(car2)

    # car3 = CarAgent("car3", file_name=input_code_name)
    # center = np.array([14.5, 3, 0, 0.6])
    # basis = np.diag([0, 0.1, 0, 0])
    # car3.set_initial(
    #     initial_state= StarSet(center, basis, C, g),
    #     initial_mode=(AgentMode.Normal, TrackMode.T0)
    # )
    # scenario.add_agent(car3)

    car4 = NPCAgent("car4")
    center = np.array([20, 3, 0, 0.5])
    basis = np.diag([0, 0.1, 0, 0])
    car4.set_initial(
        initial_state= StarSet(center, basis, C, g),
        initial_mode=(AgentMode.Normal, TrackMode.T0)
    )
    scenario.add_agent(car4)

    car5 = NPCAgent("car5")
    center = np.array([30, 0, 0, 0.5])
    basis = np.diag([0, 0.1, 0, 0])
    car5.set_initial(
        initial_state= StarSet(center, basis, C, g),
        initial_mode=(AgentMode.Normal, TrackMode.T1)
    )
    scenario.add_agent(car5)

    car6 = NPCAgent("car6")
    center = np.array([23, -3, 0, 0.5])
    basis = np.diag([0, 0.1, 0, 0])
    car6.set_initial(
        initial_state= StarSet(center, basis, C, g),
        initial_mode=(AgentMode.Normal, TrackMode.T2)
    )
    scenario.add_agent(car6)

    car7 = NPCAgent("car7")
    center = np.array([40, -6, 0, 0.5])
    basis = np.diag([0, 0.1, 0, 0])
    car7.set_initial(
        initial_state= StarSet(center, basis, C, g),
        initial_mode=(AgentMode.Normal, TrackMode.T3)
    )
    scenario.add_agent(car7)

    tmp_map = M2()
    scenario.set_map(tmp_map)
    scenario.set_sensor(BaseStarSensor())


    # bench.scenario.set_init(
    #     [
    #         [[0, -0.1, 0, 1.0], [0.0, 0.1, 0, 1.0]],
    #         [[10, -0.1, 0, 0.5], [10, 0.1, 0, 0.5]],
    #         [[14.5, 2.9, 0, 0.6], [14.5, 3.1, 0, 0.6]],
    #         [[20, 2.9, 0, 0.5], [20, 3.1, 0, 0.5]],
    #         [[30, -0.1, 0, 0.5], [30, 0.1, 0, 0.5]],
    #         [[23, -3.1, 0, 0.5], [23, -2.9, 0, 0.5]],

    #         [[40, -6.1, 0, 0.5], [40, -5.9, 0, 0.5]],
    #     ],
    #     [
    #         (AgentMode.Normal, TrackMode.T1),
    #         (AgentMode.Normal, TrackMode.T1),
    #         (AgentMode.Normal, TrackMode.T0),
    #         (AgentMode.Normal, TrackMode.T0),
    #         (AgentMode.Normal, TrackMode.T1),
    #         (AgentMode.Normal, TrackMode.T2),
    #         (AgentMode.Normal, TrackMode.T3),
    #     ],
    # )
    time_step = 0.2

    scenario.config.overwrite = False
    start = time.time()
    trace = scenario.verify(60, 0.2)
    dur = time.time()-start 
    print(f'Runtime: {dur}')
    '''
    NOTE:
    still need to finish verification, on like node 9 or something
    '''
    # plot_reachtube_stars(trace,tmp_map, filter=1)
    diams = time_step_diameter(trace, 60, 0.2)
    print(diams[-1])
    print(len(diams))
    print(sum(diams))
    # if bench.config.compare:
    #     traces1, traces2 = bench.compare_run(80, time_step)
    #     exit(0)
    # traces = bench.run(80, time_step)
    # if bench.config.plot:
    #     fig = go.Figure()
    #     fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace", sample_rate=1)
    #     fig.show()
    # if bench.config.dump:
    #     traces.dump(os.path.join(script_dir, "output3.json"))
    # bench.report()
    # start_time = time.time()
    # traces = scenario.verify(80, 0.05)
    # run_time = time.time() - start_time
    # traces.dump("./demo/tacas2023/exp3/output3.json")

    # print({
    #     "#A": len(scenario.agent_dict),
    #     "A": "C",
    #     "Map": "M2",
    #     "postCont": "DryVR",
    #     "Noisy S": "No",
    #     "# Tr": len(traces.nodes),
    #     "Run Time": run_time,
    # })

    # if len(sys.argv)>1 and sys.argv[1]=='p':
    #     fig = go.Figure()
    #     fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], 'lines', 'trace', combine_rect=3)
    #     fig.show()
