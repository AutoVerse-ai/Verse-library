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
from verse.analysis.verifier import ReachabilityMethod
from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *

if __name__ == "__main__":

    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "cardiac_cell_controller.py")
    cell = CellAgent('cell', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC

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
    scenario.config.model_path = 'cardiac_demo'
    scenario.config.model_hparams = {
        "big_initial_set": (np.array([-0.2,-0.2]), np.array([0.2,0.2])),
        "initial_set_size": 1,
    }
    
    trace = scenario.verify(5, 0.1)
    # sim = scenario.simulate(10, 0.01)
    # pp_fix(reach_at_fix(trace, 0, 10))

    stars = []
    for node in trace.nodes:
        s_mode = []
        for star in node.trace['cell']:
            s_mode.append(star)
        stars.append(s_mode)
    # plot_stars_points(stars)
    verts = []
    i = 0
    for s_mode in stars:
        v_mode = []
        # star[1].print()
        for star in s_mode:
            v_mode.append([star[0], *star[1].get_max_min(0)])
            # if i==0 and star[0]>=1.0:
            #     break
            # if i==1 and star[0]>=2.5:
            #     break
            # if i==2 and star[0]>=3.5:
            #     break
        v_mode = np.array(v_mode)
        verts.append(v_mode)
        # print([star[0], *star[1].get_max_min(0)])
#    print('Vertices:', verts)
    # verts = np.array(verts)
    #print(np.all(verts[:,2]>verts[:,1]))
    for i in range(len(verts)):
        v_mode = verts[i]
        # plt.plot(v_mode[:, 0], v_mode[:, 1], 'b.')
        # plt.plot(v_mode[:,0], v_mode[:, 2], 'r.')
        mode = 'on' if i%2==0 else 'off'
        color = 'blue' if i%2==0 else 'green'
        plt.fill_between(v_mode[:, 0], v_mode[:, 1], v_mode[:, 2], color=color, alpha=0.5, label=mode)

    plt.title('Cardiac Cell Model')
    plt.ylabel('u')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()
    # plot_reachtube_stars(trace,filter=1)

    #plt.show()