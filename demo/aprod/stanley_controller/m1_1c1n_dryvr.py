from car_true_multi_agent import CarAgent, NPCAgent
from car_true_multi_sensor import CarSensor
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import  Scenario, ScenarioConfig
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse.utils.star_diams import time_step_diameter_rect, sim_traces_to_diameters
from verse.analysis.verifier import ReachabilityMethod

import sys
import plotly.graph_objects as go
import time

class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
    M21 = auto()
    M10 = auto()

class SensorMode(Enum):
    Ready = auto()
    Update = auto()

if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "example_controller4.py")
    input_code_name_npc = os.path.join(script_dir, "controller_sensor_multi_npc.py")
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    ep_d, ep_psi = 0,0
    car_sensor = CarSensor(ep_d=ep_d, ep_psi=ep_psi) # maybe add an argument for ts
    scenario.set_sensor(car_sensor)
    base_l, base_u = [0, -0.5, 0, 1.0, 0, 0, 0, 0], [0.01, 0.5, 0, 1.0, 0, 0, 0, 0] # 4 real variables; 2 placeholders, timer, and priority
    base_l2, base_u2 = [15, -0.3, 0, 0.5, 0, 0, 0, 1], [15, 0.3, 0, 0.5, 0, 0, 0, 1] # this seems way too small consider changing later
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    
    scenario.add_agent(
        CarAgent(
            "car1",
            file_name=input_code_name,
            initial_state=[base_l, base_u],
            initial_mode=(AgentMode.Normal, TrackMode.T1, SensorMode.Ready),
        )
    )
    scenario.add_agent(
        CarAgent(
            "npc1", # use name to distinguish between car and npc in sensor -- id containing "npc" will get separate sensor logic 
            file_name=input_code_name_npc,
            initial_state=[base_l2, base_u2],
            initial_mode=(AgentMode.Normal, TrackMode.T1, SensorMode.Ready),
        )
    )
    # scenario.add_agent(NPCAgent('car3', initial_state=[[35, -3.3, 0, 0.5], [35, -2.7, 0, 0.5]], initial_mode=(AgentMode.Normal, TrackMode.T2)))
    # scenario.add_agent(NPCAgent('car4', initial_state=[[30, -0.5, 0, 0.5], [30, 0.5, 0, 0.5]], initial_mode=(AgentMode.Normal, TrackMode.T1)))
    tmp_map = M1()
    scenario.set_map(tmp_map)
    time_step, T = 1, 35 # T probably has to be at least 30 for a complete run, but check carefully
    # time_step, T = 1, 15 # T probably has to be at least 30 for a complete run, but check carefully
    
    # start_time = time.perf_counter()    
    # traces = scenario.verify_partitioned(T, time_step, 4, partition_dims=[1,2])
    # print(f'Runtime for T={T}, ts={time_step}: {time.perf_counter()-start_time:.2f}')
    # diam = time_step_diameter_rect(traces, T, time_step)
    # diam_0, diam_f, diam_bar = 1.61, diam[-1], (sum(diam)+0.0)/len(diam) # NOTE: use correct diameter values
    # print(f'F/I: {diam_f/diam_0:.5f}, A/I: {diam_bar/diam_0:.5f}\n raw final: {diam_f:.5f}, raw average: {diam_bar:.5f}, raw initial: {diam_0:.5f}')
    # fig = go.Figure()
    # fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace")
    # fig.show()

    """
    Simulations
    """
    diam_0 = 1.61
    fig = go.Figure()
    start_time = time.perf_counter()    
    N = 50
    # NOTE: simulator has the same issue that verify_refine used to have -- using previous mode to compute info, either revert change in verify_refine and talk about it in paper or fix here too
    sim_traces = scenario.simulate_multi(T, time_step, num_sims=N)
    print(f'Runtime for {N} sims, T={T}, ts={time_step}: {time.perf_counter()-start_time:.2f}')
    for sim_trace in sim_traces:
        fig = simulation_tree(sim_trace, tmp_map, fig, 1, 2, [1,2], 'lines', 'trace')

    # sim_dict = sim_traces_to_dict_composed(sim_traces)
    diam_sim = sim_traces_to_diameters(sim_traces)
    diam_f_sim, diam_bar_sim = diam_sim[-1], (sum(diam_sim)+0.0)/len(diam_sim)
    print(f'Sim results: F/I: {diam_f_sim/diam_0:.5f}, A/I: {diam_bar_sim/diam_0:.5f}\n raw final: {diam_f_sim:.5f}, raw average: {diam_bar_sim:.5f}')
    # fig.show()
    display_figure(fig)