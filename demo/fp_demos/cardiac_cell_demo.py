# import builtins
# import traceback

# original_print = builtins.print

# def debug_print(*args, **kwargs):
#     # Print the actual message
#     original_print(*args, **kwargs)
    
#     # Capture the stack, then print it using original_print (not print!)
#     stack = traceback.format_stack(limit=5)
#     original_print("---- print() called from ----")
#     for line in stack[:-1]:  # Exclude the line showing this debug_print call
#         original_print(line.strip())
#     original_print("----------------------------")

# # Override built-in print
# builtins.print = debug_print

from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go
import copy

###
from cardiac_cell_agent import CellAgent
from cardiac_cell_controller import CellMode

from z3 import *
from verse.utils.fixed_points import *


if __name__ == "__main__":

    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "cardiac_cell_controller.py")
    cell = CellAgent('cell', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC

    scenario.add_agent(cell) ### need to add breakpoint around here to check decision_logic of agents

    init_cell = [[0, 0], [0, 0]]
    # # -----------------------------------------

    scenario.set_init_single(
        'cell', init_cell,(CellMode.On,)
    )

    trace = scenario.verify(30, 0.01)
    # sim = scenario.simulate(10, 0.01)
    # pp_fix(reach_at_fix(trace, 0, 10))
    print(f'Fixed points exists? {fixed_points_fix(trace, 30, 0.01)}')

    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 0, 1, [0, 1], "fill", "trace")
    fig = reachtube_tree_slice(trace, None, fig, 0, 1, [0, 1], "fill", "trace", plot_color=colors[1:])

    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig.show()