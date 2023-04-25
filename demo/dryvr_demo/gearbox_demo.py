from origin_agent import gearbox_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.scenario import ScenarioConfig
import plotly.graph_objects as go
from enum import Enum, auto
import time

class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/sleeve_controller.py'
    config=ScenarioConfig(init_seg_length=1, parallel=False)
    scenario = Scenario(config=config)

    car = gearbox_agent('sleeve', file_name=input_code_name)
    scenario.add_agent(car)

    scenario.set_init(
        [
            [[-0.0168, 0.0029, 0, 0, 0,0,1], [-0.0166, 0.0031, 0, 0, 0,0,1]],
        ],
        [
            tuple([AgentMode.Free]),
        ]
    )
    start_time = time.time()
    traces = scenario.verify(.11, 1e-4)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Gear",
        "setup": "GRBX01",
        "result": "0",
        "time": run_time,
        "metric2": 'n/a',
        "metric3": "n/a",
    })
    traces.dump('./demo/gearbox/output.json')
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], 'lines', 'trace', sample_rate=1)
    fig.show()
