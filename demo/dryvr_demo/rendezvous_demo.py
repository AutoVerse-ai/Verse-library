from origin_agent import craft_agent
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto

import time
import plotly.graph_objects as go
from verse.plotter.plotterStar import *
from verse.utils.star_diams import *

class CraftMode(Enum):
    ProxA = auto()
    ProxB = auto()
    Passive = auto()


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/rendezvous_controller.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    car = craft_agent("test", file_name=input_code_name)
    scenario.add_agent(car)
    scenario.set_sensor(CraftSensor())
    # modify mode list input
    scenario.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.ProxA]),
        ],
    )

    start = time.time()
    traces = scenario.verify(200, 1)
    end = time.time()

    print(f'Time: {end-start}')
    diams = time_step_diameter_rect(traces, 200, 1)
    print(f'Initial diameter: {diams[0]}\n Final: {diams[-1]}\n Average: {sum(diams)/len(diams)}')


    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 3, [1, 3], "lines", "trace")
    # fig = reachtube_tree(traces, None, fig, 0, 3, [0, 3], "lines", "trace")
    fig.show()
