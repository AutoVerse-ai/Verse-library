from origin_agent_spacecraft import spacecraft_linear_agent, spacecraft_linear_agent_nd
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor
import time
import plotly.graph_objects as go
from enum import Enum, auto


class CraftMode(Enum):
    Approaching = auto()
    Rendezvous = auto()
    Aborting = auto()


if __name__ == "__main__":

    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_06.py'
    scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car4 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    scenario4.add_agent(car4)

    # modify mode list input

    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    start_time = time.time()

    traces = scenario4.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRA06",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    #fig.show()

    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_07.py'
    scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car4 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    scenario4.add_agent(car4)

    # modify mode list input

    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    start_time = time.time()

    traces = scenario4.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRA07",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    #fig.show()

    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_08.py'
    scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car4 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    scenario4.add_agent(car4)

    # modify mode list input

    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    start_time = time.time()

    traces = scenario4.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRA08",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    #fig.show()

    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_U01.py'
    scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car4 = spacecraft_linear_agent('test', file_name=input_code_name)
    scenario4.add_agent(car4)

    # modify mode list input

    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    start_time = time.time()

    traces = scenario4.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRU01",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    #fig.show()

    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_U02.py'
    scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car4 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    scenario4.add_agent(car4)

    # modify mode list input

    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    start_time = time.time()

    traces = scenario4.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRU02",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    #fig.show()
