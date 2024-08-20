from verse import Scenario, ScenarioConfig

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go
import copy

###
from cardiac_cell_agent import CellAgent
from cardiac_cell_controller import CellMode

from z3 import *
from verse.utils.fixed_points import *
from verse.analysis.verifier import ReachabilityMethod
from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *

if __name__ == "__main__":

    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "cardiac_cell_controller.py")
    cell = CellAgent('cell', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    # scenario.add_agent(cell) ### need to add breakpoint around here to check decision_logic of agents

    # init_cell = [[0, 0], [0, 0]]

    basis = np.array([[0, 0], [0, 0]]) * np.diag([0.05, 0.05]) 
    center = np.array([0,0])
    C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
    g = np.array([1,1,1,1])

    cell.set_initial(
        StarSet(center, basis, C, g),
        tuple([CellMode.On])
    )
    # # -----------------------------------------

    # scenario.set_init_single(
    #     'cell', init_cell,(CellMode.On,)
    # )
    scenario.add_agent(cell)
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    # scenario.config.pca = False
    scenario.set_sensor(BaseStarSensor())
    trace = scenario.verify(5, 0.1)
    # sim = scenario.simulate(10, 0.01)
    # pp_fix(reach_at_fix(trace, 0, 10))

    stars = []
    for star in trace.nodes[0].trace['cell']:
        stars.append(star[1])
    plot_stars_points(stars)
    print(stars[-1].basis[0][0])
    plot_reachtube_stars(trace,filter=1)
    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    plt.show()