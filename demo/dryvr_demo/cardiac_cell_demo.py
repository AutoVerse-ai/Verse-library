from verse import Scenario, ScenarioConfig

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go
import copy

###
from cardiac_cell_agent import CellAgent
from cardiac_cell_controller import CellMode

from z3 import *

from verse.analysis.verifier import ReachabilityMethod

from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *
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


    basis = np.array([[1, 0], [0, 1]]) * np.diag([0.01, 0.01]) # this doesn't actually make sense, but not sure how algorithm actually handles 1d polytopes
    center = np.array([0, 0])
    C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
    g = np.array([1,1,1,1])

    ### how do I instantiate a scenario with a starset instead of a hyperrectangle?

    cell.set_initial(
            # [[1.25, 2.25], [1.25, 2.25]],
            # [[1.55, 2.35], [1.55, 2.35]]
            StarSet(center, basis, C, g)
        ,
            tuple([CellMode.On])
            # tuple([AgentMode.Default]),
        ,
    )

    scenario.add_agent(cell)
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.set_sensor(BaseStarSensor())

    trace = scenario.verify(7, 0.01)
    ### this works, but takes around a minute