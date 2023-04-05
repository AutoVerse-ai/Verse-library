from origin_agent import gearbox_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto
import time

class CraftMode(Enum):
    Move = auto()
    Meshed = auto()

if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/gearbox_controller.py'
    #scenario = Scenario()
    scenario = Scenario()


    car = gearbox_agent('test', file_name=input_code_name)
    scenario.add_agent(car)

    # modify mode list input
    scenario.set_init(
        [
            [[0,0,-0.0168,0.0029,0,0,1], [0,0,-0.0166,0.0031,0,0,1]],
        ],
        [
            tuple([CraftMode.Move]),
        ]
    )

    # traces = scenario.simulate(3.64, .01)
    # fig = go.Figure()
    # fig = simulation_anime(traces, None, fig, 1, 2, [
    #                        1, 2], 'lines', 'trace', sample_rate=1)
    # fig.show()


    start_time = time.time()

    traces = scenario.verify(.2, 1e-6)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Gearbox",
        "setup": "GRBX01",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    fig.show()
