from origin_agent import vanderpol_agent
from verse import Scenario
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/vanderpol_controller.py"
    scenario = Scenario()

    car = vanderpol_agent("car1", file_name=input_code_name)
    scenario.add_agent(car)
    # modify mode list input
    scenario.set_init(
        [
            [[1.25, 2.25], [1.55, 2.35]],
        ],
        [
            tuple([AgentMode.Default]),
        ],
    )
    scenario.config.reachability_method = "NeuReach"
    traces = scenario.verify(
        7,
        0.05,
        params={
            "N_X0": 1,
            "N_x0": 500,
            "N_t": 100,
            "epochs": 50,
            "_lambda": 0.05,
            "use_cuda": True,
        },
    )
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2], "lines", "trace")
    fig.show()
