from car_agent_sensor import CarAgent, NPCAgent
from car_sensor import CarSensor
from verse.map import opendrive_map
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import *
from verse.analysis.verifier import ReachabilityMethod
from verse import Scenario, ScenarioConfig

import time
from enum import Enum, auto
import sys
import plotly.graph_objects as go


class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


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

class GPSMode(Enum):
    Passive = auto()
    Active = auto()

if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    # ctlr_src = "demo/vehicle/controller/intersection_car.py"
    input_code_name = os.path.join(script_dir, "controller_sensor.py")
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    # car_sensor = CarSensor()
    # scenario.set_sensor(car_sensor)

    # bench = Benchmark(sys.argv, init_seg_length=1)
    # scenario.add_agent(CarAgent("car1", file_name=input_code_name))
    # ep_d, ep_psi = 0.5, np.pi/18
    ep_d, ep_psi = 0, 0
    scenario.add_agent(CarAgent("car1", file_name=input_code_name, ep_d=ep_d, ep_psi=ep_psi))
    tmp_map = opendrive_map(os.path.join(script_dir, "t1_triple.xodr"))
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    scenario.set_map(tmp_map)
    car_sensor = CarSensor(ep_d=ep_d, ep_psi=ep_psi) # maybe add an argument for ts
    scenario.set_sensor(car_sensor)

    # base_l, base_u = [134, 11.5, 0, 5.0, 0, 0, 1], [136, 14.5, 0, 5.0, 0, 0, 1]
    base_l, base_u = [134, 11.5, 0, 5.0, 0, 0, 1], [134.01, 11.51, 0, 5.0, 0, 0, 1]
    # base_l, base_u = [136, 14.5, 0, 5.0, 0, 0, 1], [136, 14.5, 0, 5.0, 0, 0, 1]
    # init_err = [0.5, 0.5, 0, 0]
    # no_err = [0, 0,0,0]
    # init_l = base_l + [base_l[i]-init_err[i] for i in range(4)] + [-init_err[i] for i in range(4)] + [0]
    # init_u = base_u + [base_u[i]+init_err[i] for i in range(4)] + [init_err[i] for i in range(4)] + [0]

    scenario.set_init(
        [
            [base_l, base_u],
            # [init_l, init_u],
            # [[179.5, 59.5, np.pi / 2, 2.5], [180.5, 62.5, np.pi / 2, 2.5]],
            # [[124.5, 87.5, np.pi, 2.0], [125.5, 90.5, np.pi, 2.0]],
            # [[-65, -57.5, 0, 5.0], [-65, -57.5, 0, 5.0]],
            # [[15, -67.0, 0, 2.5], [15, -67.0, 0, 2.5]],
            # [[106, 18.0, 0, 2.0], [106, 18.0, 0, 2.0]],
        ],
        [
            (
                AgentMode.Normal,
                TrackMode.T1,
                # GPSMode.Passive,
            ),
            # (
            #     AgentMode.Normal,
            #     TrackMode.T1,
            # ),
            # (
            #     AgentMode.Normal,
            #     TrackMode.T2,
            # ),
        ],
    )

    time_step, T = 0.1, 20
    start_time = time.perf_counter()    
    traces = scenario.verify(T, time_step)  # traces.dump('./output1.json')
    # traces = AnalysisTree.load('./output5.json')
    print(f'Runtime for T={T}, ts={time_step}: {time.perf_counter()-start_time:.2f}')
    fig = go.Figure()
    fig = reachtube_tree(traces, tmp_map, fig, 1, 2, [1, 2], "lines", "trace")
    fig.show()
