from sleeve_agent import sleeve_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.scenario import ScenarioConfig
import plotly.graph_objects as go
from enum import Enum, auto
import sys
import time 

class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


if __name__ == "__main__":
    input_code_name = './demo/cav2023/exp12/sleeve_controller.py'
    config=ScenarioConfig(init_seg_length=1)
    scenario = Scenario(config=config)

    car = sleeve_agent('sleeve', file_name=input_code_name)
    scenario.add_agent(car)

    scenario.set_init(
        [
            [[-0.0168, 0.0029, 0, 0, 0], [-0.0166, 0.0031, 0, 0, 0]],
        ],
        [
            tuple([AgentMode.Free]),
        ]
    )
    start_time = time.time()
    traces = scenario.verify(0.1, 0.0001)
    run_time = time.time() - start_time
    # traces.dump('./demo/gearbox/output.json')
    print({
        "#A": len(scenario.agent_dict),
        "A": "G",
        "Map": "N/A",
        "postCont": "DryVR",
        "Noisy S": "N/A",
        "# Tr": len(traces.nodes),
        "Run Time": run_time,
    })
    if len(sys.argv) > 1 and sys.argv[1]=='p':        
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], 'lines', 'trace', sample_rate=1)
        fig.show()
