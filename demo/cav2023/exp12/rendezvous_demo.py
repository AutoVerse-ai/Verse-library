from origin_agent import craft_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor
import time 
import sys 

import plotly.graph_objects as go
from enum import Enum, auto


class CraftMode(Enum):
    ProxA = auto()
    ProxB = auto()
    Passive = auto()


if __name__ == "__main__":
    input_code_name = './demo/cav2023/exp12/rendezvous_controller.py'
    scenario = Scenario()

    car = craft_agent('test', file_name=input_code_name)
    scenario.add_agent(car)
    scenario.set_sensor(CraftSensor())
    # modify mode list input
    scenario.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.ProxA]),
        ]
    )
    start_time = time.time()
    traces = scenario.verify(200, 1)
    run_time = time.time() - start_time 
    print({
        "#A": len(scenario.agent_dict),
        "A": "S",
        "Map": "N/A",
        "postCont": "DryVR",
        "Noisy S": "N/A",
        "# Tr": len(traces.nodes),
        "Run Time": run_time,
    })

    if len(sys.argv) > 1 and sys.argv[1]=='p':        
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                                'lines', 'trace')
        fig.show()
