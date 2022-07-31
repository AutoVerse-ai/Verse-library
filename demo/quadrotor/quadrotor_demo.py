from quadrotor_agent import QuadrotorAgent
from verse import Scenario
from verse.plotter.plotter2D import *
import os
import json
import plotly.graph_objects as go
from enum import Enum, auto


class CraftMode(Enum):
    Follow_Waypoint = auto()


if __name__ == "__main__":
    input_code_name = './quadrotor_controller.py'
    scenario = Scenario()

    path = os.path.abspath(__file__)
    path = path.replace('quadrotor_demo.py', 'test.json')
    # print(path)
    with open(path, 'r') as f:
        prarms = json.load(f)
    waypoints = [mode[1] for mode in prarms["agents"][0]["mode_list"]]
    guard_boxes = [guard[1] for guard in prarms["agents"][0]["guards"]]
    time_limits = prarms["agents"][0]["timeHorizons"]
    # print(waypoints)
    # print(guard_boxes)
    quadrotor = QuadrotorAgent(
        'test', file_name=input_code_name, waypoints=waypoints, boxes=guard_boxes, time_limits=time_limits)
    scenario.add_agent(quadrotor)
    # scenario.set_sensor(QuadrotorSensor())
    # modify mode list input
    scenario.set_init(
        [
            [[2.75, -0.25, -0.1, 0, 0, 0, 0, 0], [3, 0, 0, 0.1, 0.1, 0.1, 0, 0]],
        ],
        [
            tuple([CraftMode.Follow_Waypoint]),
        ]
    )

    traces = scenario.simulate(200, 0.2)
    fig = go.Figure()
    fig = simulation_tree(traces, None, fig, 1, 2,
                          'lines', 'trace', print_dim_list=[0, 1, 2])
    fig = fig.add_trace(go.Scatter(
        x=[3, 5, 5, 2, 2, 8, 8], y=[0, 0, 3, 3, 6, 3, 0], text=[0, 1, 2, 3, 4, 5, 6], mode='markers', marker={'color': 'black'}))
    fig.show()


    traces = scenario.verify(200, 0.2)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2,
                          'lines', 'trace', print_dim_list=[0, 1, 2])
    fig = fig.add_trace(go.Scatter(
        x=[3, 5, 5, 2, 2, 8, 8], y=[0, 0, 3, 3, 6, 3, 0], text=[0, 1, 2, 3, 4, 5, 6], mode='markers', marker={'color': 'black'}))
    fig.show()
