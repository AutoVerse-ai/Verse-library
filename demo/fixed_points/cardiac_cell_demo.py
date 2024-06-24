from verse import Scenario, ScenarioConfig
from vehicle_controller import VehicleMode, TLMode

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go
import copy

###
from cardiac_cell_agent import CellAgent
from cardiac_cell_controller import CellMode

from z3 import *
from fixed_points import fixed_points_aa_branching, fixed_points_aa_branching_composed, contained_single, reach_at, fixed_points_sat, reach_at_fix, fixed_points_fix
from fixed_points import contain_all_fix, contain_all, pp_fix, pp_old

if __name__ == "__main__":

    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "cardiac_cell_controller.py")
    cell = CellAgent('cell', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(cell) ### need to add breakpoint around here to check decision_logic of agents

    init_cell = [[0, 0], [0, 0]]
    # # -----------------------------------------

    scenario.set_init_single(
        'cell', init_cell,(CellMode.On,)
    )

    trace = scenario.verify(10, 0.01)
    sim = scenario.simulate(10, 0.01)
    pp_fix(reach_at_fix(trace, 0, 10))
    print(f'Fixed points exists? {fixed_points_fix(trace, 15, 0.01)}')

    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 0, 1, [0, 1], "fill", "trace")
    fig = reachtube_tree(trace, None, fig, 0, 1, [0, 1], "fill", "trace")

    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig.show()