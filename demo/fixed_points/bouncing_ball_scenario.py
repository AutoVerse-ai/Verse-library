from verse import Scenario, ScenarioConfig
from vehicle_controller import VehicleMode, TLMode

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go
import copy

###
from bouncing_ball import BallAgent
from ball_controller import BallMode

from z3 import *
from fixed_points import fixed_points_aa_branching, fixed_points_aa_branching_composed, contained_single, reach_at, fixed_points_sat, reach_at_fix, fixed_points_fix
from fixed_points import contain_all_fix, contain_all, pp_fix, pp_old

if __name__ == "__main__":

    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "ball_controller.py")
    ball = BallAgent('ball', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(ball) ### need to add breakpoint around here to check decision_logic of agents

    init_ball = [[10,2],[10,2]]
    # # -----------------------------------------

    scenario.set_init_single(
        'ball', init_ball,(BallMode.Normal,)
    )

    trace = scenario.verify(7, 0.01)

    pp_fix(reach_at_fix(trace, 0, 7))
    print(f'Fixed points exists? {fixed_points_fix(trace)}')

    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 0, 1, [0, 1], "fill", "trace")
    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig.show()