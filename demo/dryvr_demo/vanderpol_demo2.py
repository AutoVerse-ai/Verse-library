from origin_agent import vanderpol_agent
from verse import Scenario,ScenarioConfig
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto

### this is a hack
import sys
import os

neighbor_directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fixed_points'))
sys.path.append(neighbor_directory_path)

from verse.utils.fixed_points import *
###

class AgentMode(Enum):
    Default = auto()

if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "vanderpol_controller.py")
    scenario = Scenario(ScenarioConfig(parallel=False))

    car = vanderpol_agent("car1", file_name=input_code_name) # fix name
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
    # scenario.config.reachability_method = "NeuReach"
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
    fig = reachtube_tree_slice(traces, None, fig, 1, 2, [1, 2], "lines", "trace", plot_color=colors[1:])
    for i in range(10):
        sim = scenario.simulate(7, 0.05)
        fig = simulation_tree(sim, None, fig, 1, 2, [1, 2], "lines", "trace", plot_color=colors[2:])
    fig.show()
    print("last", reach_at_fix(traces))
    print("Do fixed points exist for this scenario: ", fixed_points_fix(traces, 7, 0.05))
    # pp_fix(reach_at_fix(traces, 0, 6.96))
    # pp_fix(reach_at_fix(traces))
    ### show the actual trajectories for t>7
    ### superimpose the final slice
    ### fixed points works for this