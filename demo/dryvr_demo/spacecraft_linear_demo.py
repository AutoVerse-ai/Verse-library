from origin_agent_spacecrafft import spacecraft_linear_agent
from verse import Scenario
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
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller.py'
    scenario = Scenario()

    car = spacecraft_linear_agent('test', file_name=input_code_name)
    scenario.add_agent(car)

    # modify mode list input

    scenario.set_init(
        [
            [[-925, -425, 0, 0,0,0], [-875, -375, 0, 0, 0,0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    # traces = scenario.simulate(200, 1)
    # fig = go.Figure()
    # fig = simulation_anime(traces, None, fig, 1, 2, [
    #                        1, 2], 'lines', 'trace', sample_rate=1)
    # fig.show()


    start_time = time.time()

    traces = scenario.verify(300, 1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Spacecraft",
        "setup": "SRA01",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    fig.show()

    input_code_name = './demo/dryvr_demo/spacecraft_linear_controller2.py'
    scenario1 = Scenario()
    car = spacecraft_linear_agent('test', file_name=input_code_name)
    scenario.add_agent(car)

    # modify mode list input

    scenario1.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )



    start_time = time.time()

    traces = scenario1.verify(300, 1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Spacecraft",
        "setup": "SRA01",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
