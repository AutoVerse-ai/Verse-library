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
    Normal = auto()
    Switch_Down = auto()
    Switch_Up = auto()
    Switch_Left = auto()
    Switch_Right = auto()


class TrackMode(Enum):
    Lane0 = auto()
    Lane1 = auto()
    Lane2 = auto()


if __name__ == "__main__":
    input_code_name = './demo/quadrotor/quadrotor_controller2.py'
    scenario = Scenario()

    path = os.path.abspath(__file__)
    path = path.replace('demo3.py', 'test.json')
    # print(path)
    with open(path, 'r') as f:
        prarms = json.load(f)
    time_step = 0.1
    # quadrotor1 = QuadrotorAgent('test1', file_name=input_code_name)
    # scenario.add_agent(quadrotor1)
    quadrotor2 = QuadrotorAgent('test2', file_name=input_code_name)
    scenario.add_agent(quadrotor2)
    t_v = {  # quadrotor1.id: (1, 1),
        quadrotor2.id: (1, 1)}
    bs = {  # quadrotor1.id: [0.4, 0.4, 0.4, 20, 20, 20],
        quadrotor2.id: [0.4, 0.4, 0.4]}
    tmp_map = SimpleMap3(t_v_pair=t_v, box_side=bs)
    scenario.set_map(tmp_map)
    # fig = go.Figure()
    # draw_map_3d(map=tmp_map, fig=fig)
    scenario.set_sensor(QuadrotorSensor())
    # modify mode list input
    init_l = [-15, -4, 1, 0, 0, 0]
    init_u = [-14, -3, 1, 0, 0, 0]
    scenario.set_init(
        [
            #[[2.75, 0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0.1, 0.1, 0.1, 0, 0]],
            [init_l, init_u],
            # [[4, -4.5, -0.1, 0, 0, 0, 0, 0], [4, -4.5, -0.1, 0, 0, 0, 0, 0]],
        ],
        [
            #tuple([CraftMode.Follow_Lane, TrackMode.Lane1]),
            tuple([CraftMode.Normal, TrackMode.Lane1])
        ]
    )
    traces = scenario.simulate(200, time_step)
    # path = os.path.abspath(__file__)
    # path = path.replace('quadrotor_demo2.py', 'output_2.json')
    # write_json(traces, path)
    fig = go.Figure()
    fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [0, 1, 2, 3],
                             'lines', 'trace')
    fig.show()

    # traces = scenario.simulate(30, time_step)
    # fig = go.Figure()
    # fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [0, 1, 2, 3],
    #                          'lines', 'trace')
    # fig.show()
