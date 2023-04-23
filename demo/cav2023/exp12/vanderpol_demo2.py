from origin_agent import vanderpol_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
import time 
import  sys 

import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()


if __name__ == "__main__":
    input_code_name = './demo/cav2023/exp12/vanderpol_controller.py'
    scenario = Scenario()

    car = vanderpol_agent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    # modify mode list input
    scenario.set_init(
        [
            [[1.25, 2.25], [1.55, 2.35]],
        ],
        [
            tuple([AgentMode.Default]),
        ]
    )
    start_time = time.time()
    traces = scenario.verify(
        7, 0.01,
    )
    run_time = time.time() - start_time 
    print({
        "#A": len(scenario.agent_dict),
        "A": "V",
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
