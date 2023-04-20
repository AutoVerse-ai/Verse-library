from origin_agent import spacecraft_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto
import time

class CraftMode(Enum):
    Approaching = auto()
    Rendezvous = auto()
    Aborting = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/spacecraft_controller.py'
    scenario = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))

    car = spacecraft_agent('test', file_name=input_code_name)
    scenario.add_agent(car)

    # modify mode list input
    scenario.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 5, 5, 0, 0]],
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

    traces = scenario.verify(200, 0.05, params = {"sim_trace_num":100})
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "SPRE22",
        "setup": "",
        "result": "1",
        "time": run_time,
        "metric2": "",
        "metric3": "",
    })
    # traces.dump('output.json')
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
    #                      'lines', 'trace','true')

    # fig.show()
