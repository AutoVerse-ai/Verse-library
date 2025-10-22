from car_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M2
# from verse.scenario.scenario import Benchmark
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from car_sensor_2c5n import CarSensor

from enum import Enum, auto
import sys
import plotly.graph_objects as go


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

class GPSMode(Enum):
    Passive = auto()
    Active = auto()

if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "example_controller7.py")

    # input_code_name = "./demo/tacas2023/exp3/example_controller7.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    car_sensor = CarSensor()
    scenario.set_sensor(car_sensor)

    car = CarAgent("car1", file_name=input_code_name)
    scenario.add_agent(car)
    car = NPCAgent("car2")
    scenario.add_agent(car)
    car = CarAgent("car3", file_name=input_code_name)
    scenario.add_agent(car)
    car = NPCAgent("car4")
    scenario.add_agent(car)
    car = NPCAgent("car5")
    scenario.add_agent(car)
    car = NPCAgent("car6")
    scenario.add_agent(car)
    car = NPCAgent("car7")
    scenario.add_agent(car)
    tmp_map = M2()
    scenario.set_map(tmp_map)
    
    base_l, base_u = [0, -0.1, 0, 1.0], [0.0, 0.1, 0, 1.0]
    init_err = [1,1,0,0]
    no_err = [0,0,0,0]
    car1_l = base_l + [base_l[i]-init_err[i] for i in range(4)] + [-init_err[i] for i in range(4)] + [0]
    car1_u = base_u + [base_u[i]+init_err[i] for i in range(4)] + [init_err[i] for i in range(4)] + [0]

    base_3_l, base_3_u = [14.5, 2.9, 0, 0.6], [14.5, 3.1, 0, 0.6]
    car3_l = base_3_l + [base_3_l[i]-init_err[i] for i in range(4)] + [-init_err[i] for i in range(4)] + [1]
    car3_u = base_3_u + [base_3_u[i]+init_err[i] for i in range(4)] + [init_err[i] for i in range(4)] + [1] 


    scenario.set_init(
        [
            [car1_l, car1_u],
            # [[0, -0.1, 0, 1.0], [0.0, 0.1, 0, 1.0]],
            [[10, -0.1, 0, 0.5], [10, 0.1, 0, 0.5]],
            [car3_l, car3_u],
            # [[14.5, 2.9, 0, 0.6], [14.5, 3.1, 0, 0.6]],
            [[20, 2.9, 0, 0.5], [20, 3.1, 0, 0.5]],
            [[30, -0.1, 0, 0.5], [30, 0.1, 0, 0.5]],
            [[23, -3.1, 0, 0.5], [23, -2.9, 0, 0.5]],
            [[40, -6.1, 0, 0.5], [40, -5.9, 0, 0.5]],
        ],
        [
            (AgentMode.Normal, TrackMode.T1, GPSMode.Passive),
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T0, GPSMode.Passive),
            (AgentMode.Normal, TrackMode.T0),
            (AgentMode.Normal, TrackMode.T1),
            (AgentMode.Normal, TrackMode.T2),
            (AgentMode.Normal, TrackMode.T3),
        ],
    )
    time_step = 0.05

    traces = scenario.verify(40, time_step)
    fig = go.Figure()
    fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace")
    fig.update_layout(
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        # legend_title='Trajectory Types',
    )
    fig.show()
