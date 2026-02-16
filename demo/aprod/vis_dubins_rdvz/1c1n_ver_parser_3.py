from car_agent import CarAgent, NPCAgent
from car_sensor_parser_3 import CarSensor
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from verse.analysis.verifier import ReachabilityMethod
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse import Scenario, ScenarioConfig
from verse.utils.star_diams import time_step_diameter_rect, sim_traces_to_dict_composed, sim_traces_to_diameters
import time

import sys
import plotly.graph_objects as go
from parser_wrapper import clear_parse_cache

class AgentMode(Enum):
    Normal = auto()
    Left = auto()
    Right = auto()

class AssignMode(Enum):
    Assigned = auto()
    Waiting = auto()
    Complete = auto()

if __name__ == "__main__":
    import os

    clear_parse_cache()
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "controller_3.py")
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    car_sensor = CarSensor()
    scenario.set_sensor(car_sensor)
    base_l, base_u = [0, 0, 0, 1.0], [0.01, 0.01, 0, 1.0]
    # base_l, base_u = [0, 0, 0, 1.0], [0.1, 0.1, 0, 1.0]
    # init_err = [1, 1, 0, 0]
    no_err = [0, 0,0,0]
    scenario.add_agent(
        CarAgent(
            "car1",
            file_name=input_code_name, 
            # structure is x,hx,ex,t
            # initial_state=[[0, -0.5, 0, 1.0]+ [base_l[i]-init_err[i] for i in range(4)] + [-0.1, -0.1, 0, -0.01] + [0],  
                        #    [0.01, 0.5, 0, 1.0]+[base_u[i]+init_err[i] for i in range(4)]+ init_err + [0]],
            initial_state= [base_l+base_l+no_err+[0, 0, 1, 1, 0, -1, 0], # structure is timer, _id, connected_ids, assigned_id, dist, prev_sense, and cur_sense
                            base_u+base_u+no_err+[0, 0, 1, 1, 0, -1, 0]] # just assume that the cars know that they start in -1
                             ,
            initial_mode=(AgentMode.Normal, AssignMode.Assigned),
        )
    )

    base_l_2, base_u_2 = [15, 2, np.pi, 1], [15.01, 2.01, np.pi, 1]
    # base_l_2, base_u_2 = [15, 2, np.pi, 1], [15.1, 2.1, np.pi, 1]
    scenario.add_agent(
        CarAgent(
            "car2",
            file_name=input_code_name, 
            initial_state=[base_l_2+base_l_2+no_err+[0, 1, 2, 0, 0, -1, 0],
                            base_u_2+base_u_2+no_err+[0, 1, 2, 0, 0, -1, 0]]
                            ,
            initial_mode=(AgentMode.Normal, AssignMode.Assigned),
        )
    )

    base_l_3, base_u_3 = [25, 10, -7*np.pi/8, 1], [25, 10, -7*np.pi/8, 1] # no uncertainty for now -- issue where 60, -30 wmean
    scenario.add_agent(
        CarAgent(
            "car3",
            file_name=input_code_name, 
            initial_state=[base_l_3+base_l_3+no_err+[0, 2, 4, 0, 0, -1, 0],
                            base_u_3+base_u_3+no_err+[0, 2, 4, 0, 0, -1, 0]]
                            ,
            initial_mode=(AgentMode.Normal, AssignMode.Assigned),
        )
    )

    # T, time_step = 7.5, 0.05
    T, time_step = 15, 0.05


    # start = time.perf_counter()
    # traces = scenario.verify(T, time_step)
    # print(f'Runtime for T={T}, ts={time_step}: {time.perf_counter()-start:.2f}')
    # diam = time_step_diameter_rect(traces, T, time_step)
    # diam_0, diam_f, diam_bar = 0.4, diam[-1], (sum(diam)+0.0)/len(diam)
    # # diam_0, diam_f, diam_bar = 0.04, diam[-1], (sum(diam)+0.0)/len(diam)
    # print(f'F/I: {diam_f/diam_0:.2f}, A/I: {diam_bar/diam_0:.2f}; raw final: {diam_f:.2f}, raw average: {diam_bar:.2f}')
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2], "lines", "trace")
    # # fig = reachtube_tree(traces, fig, 1, 2, [1, 2], "lines", "trace")
    # fig.update_layout(
    #     xaxis_title='x (m)',
    #     yaxis_title='y (m)',
    #     # legend_title='Trajectory Types',
    # )
    # fig.show()

    """Sim"""
    start_time = time.perf_counter()    
    N = 25
    sim_traces = scenario.simulate_multi(T, time_step, num_sims=N)
    print(f'Runtime for {N} sims, T={T}, ts={time_step}: {time.perf_counter()-start_time:.2f}')
    fig = go.Figure()
    for sim_trace in sim_traces:
        fig = simulation_tree(sim_trace, None, fig, 1, 2, [0,1,2], 'lines', 'trace')

    # sim_dict = sim_traces_to_dict_composed(sim_traces)
    # fig.show()
    display_figure(fig)
    diam_0 = 0.04
    diam_sim = sim_traces_to_diameters(sim_traces)
    diam_f_sim, diam_bar_sim = diam_sim[-1], (sum(diam_sim)+0.0)/len(diam_sim)
    print(f'Sim results: F/I: {diam_f_sim/diam_0:.5f}, A/I: {diam_bar_sim/diam_0:.5f}\n raw final: {diam_f_sim:.5f}, raw average: {diam_bar_sim:.5f}')
