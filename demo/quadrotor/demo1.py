from quadrotor_agent import QuadrotorAgent
from verse import Scenario
from verse.map.lane_3d import Lane_3d
from verse.map.example_map.simple_map_3d import SimpleMap1
from verse.plotter.plotter3D_new import *
from verse.sensor.example_sensor.quadrotor_sensor2 import QuadrotorSensor
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
    Lane3 = auto()
    Lane4 = auto()


if __name__ == "__main__":
    input_code_name = './demo/quadrotor/quadrotor_controller.py'
    scenario = Scenario()

    path = os.path.abspath(__file__)
    path = path.replace('demo1.py', 'test.json')
    # print(path)
    with open(path, 'r') as f:
        prarms = json.load(f)
    time_step = 0.05
    quadrotor1 = QuadrotorAgent(
        'test1', file_name=input_code_name)
    scenario.add_agent(quadrotor1)
    quadrotor2 = QuadrotorAgent(
        'test2', file_name=input_code_name)
    scenario.add_agent(quadrotor2)
    t_v = {quadrotor1.id: (1, 2),
           quadrotor2.id: (1, 1)
           }
    bs = {quadrotor1.id: [0.4, 0.4, 0.4],
          quadrotor2.id: [0.4, 0.4, 0.4]
          }
    tmp_map = SimpleMap1(t_v_pair=t_v, box_side=bs)
    scenario.set_map(tmp_map)
    scenario.set_sensor(QuadrotorSensor())
    # modify mode list input
    scenario.set_init(
        [
            [[2.75, 0, 0, 0, 0,  0], [3, 0, 0, 0.1, 0.1, 0.1]],
            [[5, 0, 0, 0, 0, 0], [10, 0, 0, 0.1, 0.1, 0.1]],
        ],
        [
            tuple([CraftMode.Normal, TrackMode.Lane0]),
            tuple([CraftMode.Normal, TrackMode.Lane0])
        ]
    )
    traces = scenario.simulate(20, time_step)
    path = os.path.abspath(__file__)
    path = path.replace('quadrotor_demo.py', 'output.json')
    write_json(traces, path)
    fig = go.Figure()
    fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [0, 1, 2, 3],
                             'lines', 'trace')
    # fig = fig.add_trace(go.Scatter(
    #     x=[3, 5, 5, 2, 2, 8, 8], y=[0, 0, 3, 3, 6, 3, 0], text=[0, 1, 2, 3, 4, 5, 6], mode='markers', marker={'color': 'black'}))
    fig.show()

    # traces = scenario.verify(40, time_step)
    # fig = go.Figure()
    # fig = reachtube_tree_3d(traces, tmp_map, fig, 1, 2, 3, [0, 1, 2, 3],
    #                         'lines', 'trace')
    # fig.show()
