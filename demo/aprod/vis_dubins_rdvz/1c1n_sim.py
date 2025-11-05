from car_agent import CarAgent, NPCAgent
from car_sensor import CarSensor
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from verse.analysis.verifier import ReachabilityMethod
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse import Scenario, ScenarioConfig

import sys
import plotly.graph_objects as go


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

    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "controller.py")
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    car_sensor = CarSensor()
    scenario.set_sensor(car_sensor)
    base_l, base_u = [0, 0, 0, 1.0], 0
    # init_err = [1, 1, 0, 0]
    no_err = [0, 0,0,0]
    scenario.add_agent(
        CarAgent(
            "car1",
            file_name=input_code_name, 
            # structure is x,hx,ex,t
            # initial_state=[[0, -0.5, 0, 1.0]+ [base_l[i]-init_err[i] for i in range(4)] + [-0.1, -0.1, 0, -0.01] + [0],  
                        #    [0.01, 0.5, 0, 1.0]+[base_u[i]+init_err[i] for i in range(4)]+ init_err + [0]],
            initial_state= [base_l+base_l+no_err+[0, 0, 1, 1, 0, -3, 0], # structure is timer, _id, connected_ids, assigned_id, dist, prev_sense, and cur_sense
                            base_l+base_l+no_err+[0, 0, 1, 1, 0, -3, 0]] # -2 is a sentinel value 
                             ,
            initial_mode=(AgentMode.Normal, AssignMode.Assigned),
        )
    )

    base_l_2 = [15, 2, np.pi, 1]
    scenario.add_agent(
        CarAgent(
            "car2",
            file_name=input_code_name, 
            initial_state=[base_l_2+base_l_2+no_err+[0, 1, 2, 0, 0, -3, 0],
                            base_l_2+base_l_2+no_err+[0, 1, 2, 0, 0, -3, 0]]
                            ,
            initial_mode=(AgentMode.Normal, AssignMode.Assigned),
        )
    )
    time_step = 0.1

    traces = scenario.simulate(10, time_step)
    fig = go.Figure()
    fig = simulation_tree(traces, None, fig, 1, 2, [1, 2], "lines", "trace")
    # fig = reachtube_tree(traces, fig, 1, 2, [1, 2], "lines", "trace")
    fig.update_layout(
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        # legend_title='Trajectory Types',
    )
    fig.show()
