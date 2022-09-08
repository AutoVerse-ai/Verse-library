from audioop import tomono
from quadrotor_agent import QuadrotorAgent
from verse import Scenario
# from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
from verse.map.example_map.simple_map_3d import SimpleMap1, SimpleMap2, SimpleMap3
from verse.sensor.example_sensor.quadrotor_sensor import QuadrotorSensor
import os
import json
import plotly.graph_objects as go
from enum import Enum, auto
from gen_json import write_json, read_json


class CraftMode(Enum):
    Follow_Waypoint = auto()
    Follow_Lane = auto()


class LaneMode(Enum):
    Lane0 = auto()
    Lane1 = auto()
    Lane2 = auto()


if __name__ == "__main__":
    input_code_name = './demo/quadrotor/quadrotor_controller.py'
    scenario = Scenario()

    path = os.path.abspath(__file__)
    path = path.replace('quadrotor_demo2.py', 'test.json')
    # print(path)
    with open(path, 'r') as f:
        prarms = json.load(f)
    waypoints = [mode[1] for mode in prarms["agents"][0]["mode_list"]]
    guard_boxes = [guard[1] for guard in prarms["agents"][0]["guards"]]
    time_limits = prarms["agents"][0]["timeHorizons"]
    # print(waypoints)
    # print(guard_boxes)
    time_step = 0.05
    # quadrotor1 = QuadrotorAgent(
    #     'test1', file_name=input_code_name, time_limits=[1]*100)
    # scenario.add_agent(quadrotor1)
    quadrotor2 = QuadrotorAgent(
        'test2', file_name=input_code_name, time_limits=[1]*100)
    scenario.add_agent(quadrotor2)
    wps = {  # quadrotor1.id: [[3, 0, 0, 4, 0, 0]],
        quadrotor2.id: [[0, 0, 0]]}
    t_v = {  # quadrotor1.id: (1, 1),
        quadrotor2.id: (1, 1)}
    bs = {  # quadrotor1.id: [0.4, 0.4, 0.4, 20, 20, 20],
        quadrotor2.id: [0.4, 0.4, 0.4, 20, 20, 20]}
    tmp_map = SimpleMap2(waypoints=wps, t_v_pair=t_v, box_side=bs)
    scenario.set_map(tmp_map)
    scenario.set_sensor(QuadrotorSensor())
    # modify mode list input
    scenario.set_init(
        [
            #[[2.75, 0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0.1, 0.1, 0.1, 0, 0]],
            [[4, -4.5, -0.1, 0, 0, 0, 0, 0], [4.2, -4.3, 0, 0.1, 0.1, 0.1, 0, 0]],
            # [[4, -4.5, -0.1, 0, 0, 0, 0, 0], [4, -4.5, -0.1, 0, 0, 0, 0, 0]],
        ],
        [
            #tuple([CraftMode.Follow_Lane, LaneMode.Lane1]),
            tuple([CraftMode.Follow_Lane, LaneMode.Lane1])
        ]
    )
    traces = scenario.verify(30, time_step)
    path = os.path.abspath(__file__)
    path = path.replace('quadrotor_demo2.py', 'output_2.json')
    write_json(traces, path)
    fig = go.Figure()
    fig = reachtube_tree_3d(traces, tmp_map, fig, 1, 2, 3, [0, 1, 2, 3],
                            'lines', 'trace')
    fig.show()
    # traces = scenario.simulate(50, time_step)
    # fig = go.Figure()
    # fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [0, 1, 2, 3],
    #                          'lines', 'trace')
    # fig.show()
