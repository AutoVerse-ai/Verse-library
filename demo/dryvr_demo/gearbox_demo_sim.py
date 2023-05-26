from sleeve_agent import sleeve_agent
from verse import Scenario
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/sleeve_controller.py"
    scenario = Scenario()

    car = sleeve_agent("sleeve", file_name=input_code_name)
    scenario.add_agent(car)

    scenario.set_init(
        [
            [[-0.0168, 0.0029, 0, 0, 0], [-0.0166, 0.0031, 0, 0, 0]],
        ],
        [
            tuple([AgentMode.Free]),
        ],
    )
    traces = scenario.simulate(0.2, 0.0001)
    # traces.dump('./demo/gearbox/output.json')
    fig = go.Figure()
    fig = simulation_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], "lines", "trace")
    fig.show()
