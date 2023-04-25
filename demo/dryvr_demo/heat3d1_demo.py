from origin_agent_heat import heat3d1_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto
import time

class CraftMode(Enum):
    Default = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/heat3d1_controller.py'
    #scenario = Scenario()
    scenario = Scenario()


    car = heat3d1_agent('test', file_name=input_code_name)
    scenario.add_agent(car)

    # modify mode list input
    scenario.set_init(
        [
            [[.9,.9,.9,0,0,.9,.9,.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.9,.9,.9,0,0,.9,.9,.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1.1,1.1,1.1,0,0,1.1,1.1,1.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.1,1.1,1.1,0,0,1.1,1.1,1.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        ],
        [
            tuple([CraftMode.Default]),
        ]
    )


    start_time = time.time()

    traces = scenario.verify(40, 0.02, params = {"sim_trace_num":100})
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Heat3D",
        "setup": "HEAT01",
        "result": "1",
        "time": run_time,
        "metric2" : 'n/a',
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 0, 63, [0, 63],
                         'lines', 'trace')
    # fig.add_trace(go.Scatter(x=[
    #         1+0.157,1+0.033,1+-0.033,1+-0.157,1+-0.157,1+-0.033,1+0.033,1+0.157,1+0.157
    #     ],
    #     y=[
    #         1+0.033,1+0.157,1+0.157,1+0.033,1+-0.033,1+-0.157,1+-0.157,1+-0.033,1+0.033
    #     ]
    # ))
    fig.update_layout(
        xaxis_title="t", yaxis_title="x62"
    )
    fig.show()
