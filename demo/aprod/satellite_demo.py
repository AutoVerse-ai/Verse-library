from satellite_agent import SatelliteAgent
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D import *

import plotly.graph_objects as go
from enum import Enum, auto


class SatelliteMode(Enum):
    Passive = auto()


if __name__ == "__main__":
    input_code_name = "./demo/aprod/satellite_controller.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    dep = SatelliteAgent("deputy", file_name=input_code_name)
    scenario.add_agent(dep)
    # scenario.set_sensor(CraftSensor())
    # modify mode list input
    scenario.set_init(
        [
            [[-925, -425, 0, 0, 0, 0, -925, -425, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
             [-875, -375, 0, 0, 0, 0, -875, -375, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
        ],
        [
            (SatelliteMode.Passive,)
        ],
    )

    N = 10
    traces = []
    for i in range(N):
        trace = scenario.simulate(10, 0.1)
        traces.append(trace)

    fig = go.Figure()
    for trace in traces:
        fig = simulation_tree(trace, None, fig, 0, 7, [0, 7], "lines", "trace")
        fig = simulation_tree(trace, None, fig, 0, 1, [0, 1], "lines", "trace")
    fig.show()
