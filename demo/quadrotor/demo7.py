from quadrotor_agent2 import QuadrotorAgent
from verse import Scenario
from verse.plotter.plotter2D import simulation_tree
from verse.plotter.plotter3D_new import *
from verse.plotter.plotter3D import *
from verse.map.example_map.simple_map_3d import SimpleMap6
from verse.sensor.example_sensor.quadrotor_sensor import QuadrotorSensor
import os
import json
import plotly.graph_objects as go
import pyvista as pv 
from enum import Enum, auto
from gen_json import write_json, read_json


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


if __name__ == "__main__":
    input_code_name = './demo/quadrotor/quadrotor_controller5.py'
    input_code_name2 = './demo/quadrotor/quadrotor_controller6.py'
    scenario = Scenario()

    # path = os.path.abspath(__file__)
    # path = path.replace('demo7.py', 'test.json')
    # # print(path)
    # with open(path, 'r') as f:
    #     prarms = json.load(f)
    time_step = 0.1
    quadrotor1 = QuadrotorAgent('test1', file_name=input_code_name)
    scenario.add_agent(quadrotor1)
    init_l_1 = [9.5, 0,0,0, 0, 0,0,0,-0.3, 0]
    init_u_1 = [10,0,0,0, 0.5,0,0,0, 0.3, 0]
    # init_u_1 = [3.5, -8.5, -1, 0, 0, 0]
    scenario.set_init_single(quadrotor1.id, [init_l_1, init_u_1], tuple(
        [CraftMode.Normal, TrackMode.T1]))

    quadrotor2 = QuadrotorAgent('test2', file_name=input_code_name2)
    scenario.add_agent(quadrotor2)
    init_l_2 = [3,0,0,0, 9,0,0,0, -0.3, 0]
    init_u_2 = [3.5,0,0,0, 9.5,0,0,0, 0.3, 0]
    # init_u_2 = [3, -5, -4, 0, 0, 0]
    scenario.set_init_single(quadrotor2.id, [init_l_2, init_u_2], tuple(
        [CraftMode.Normal, TrackMode.T1]))

    # quadrotor3 = QuadrotorAgent('test3', file_name=input_code_name2)
    # scenario.add_agent(quadrotor3)
    # init_l_3 = [3, 9, 7.7, 0, 0, 0]
    # init_u_3 = [3.5, 9.5, 8.3, 0, 0, 0]
    # # init_u_2 = [3, -5, -4, 0, 0, 0]
    # scenario.set_init_single(quadrotor3.id, [init_l_3, init_u_3], tuple(
    #     [CraftMode.Normal, TrackMode.T0]))

    t_v = {quadrotor1.id: (1, 0.3),
           quadrotor2.id: (1, 0.3),
        #    quadrotor3.id: (1, 0.85),
           }
    bs = {quadrotor1.id: [0.4]*3,
          quadrotor2.id: [0.4]*3,
        #   quadrotor3.id: [0.4]*3
          }
    tmp_map = SimpleMap6(t_v_pair=t_v, box_side=bs)
    scenario.set_map(tmp_map)
    fig = go.Figure()
    # draw_map_3d(tmp_map, fig, 'lines')
    # scenario.set_sensor(QuadrotorSensor())

    # path = os.path.abspath(__file__)
    # path = path.replace('demo6.py', 'output_2.json')
    traces = scenario.simulate(600, time_step)
    # write_json(traces, path)
    fig = go.Figure()
    # traces = read_json(path)
    fig = simulation_tree_3d(traces, tmp_map, fig, 1, 5, 9, [0, 1, 5, 9],
                             'lines', 'trace')
    # fig = simulation_tree(traces, map = None, fig = fig, x_dim = 0, y_dim = 1)
    # fig = simulation_tree(traces, map = None, fig = fig, x_dim = 0, y_dim = 5)
    fig.show()


    # traces = scenario.verify(40, time_step)
    # fig = pv.Plotter()
    # fig = plot3dReachtube(traces, 'test1',1,2,3,'r',fig)
    # fig = plot3dReachtube(traces, 'test2',1,2,3,'g',fig)
    # fig.show()
