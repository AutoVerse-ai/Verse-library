from sleeve_agent import sleeve_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.scenario import ScenarioConfig
import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/sleeve_controller2.py"
    config = ScenarioConfig(init_seg_length=1)
    scenario = Scenario(config=config)

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
    traces = scenario.verify(0.2, 0.00001)
    traces.dump("./demo/gearbox/output.json")
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], "lines", "trace", sample_rate=1)
    fig.show()
