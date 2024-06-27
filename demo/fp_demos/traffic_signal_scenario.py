from mp4_p2 import VehicleAgent, TrafficSignalAgent, TrafficSensor, eval_velocity, sample_init
from verse import Scenario, ScenarioConfig
from vehicle_controller import VehicleMode, TLMode

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
from verse.utils.fixed_points import *
import plotly.graph_objects as go
import copy

###

from ball_scenario_copy import BallScenario
from ball_scenario_branch import BallScenarioBranch
from ball_scenario_branch_nt import BallScenarioBranchNT
from z3 import *

from reachtube_copy import reachtube_tree_slice

if __name__ == "__main__":

    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "vehicle_controller.py")
    vehicle = VehicleAgent('car', file_name=input_code_name)
    input_code_name = os.path.join(script_dir, "traffic_controller.py")
    tl = TrafficSignalAgent('tl', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(vehicle) ### need to add breakpoint around here to check decision_logic of agents
    scenario.add_agent(tl)
    scenario.set_sensor(TrafficSensor())

    # # ----------- Different initial ranges -------------
    # # Uncomment this block to use R1
    # init_car = [[0,-5,0,5],[50,5,0,5]]
    # init_trfficlight = [[300,0,0,0,0],[300,0,0,0,0]]
    # # -----------------------------------------

    # # Uncomment this block to use R2
    # init_car = [[0,-5,0,5],[100,5,0,5]]
    # init_trfficlight = [[300,0,0,0,0],[300,0,0,0,0]]
    # # -----------------------------------------

    # # Uncomment this block to use R3
    init_car = [[0,-5,0,0],[100,5,0,10]]
    init_trfficlight = [[300,0,0,0,0],[300,0,0,0,0]]
    # # -----------------------------------------

    scenario.set_init_single(
        'car', init_car,(VehicleMode.Normal,)
    )
    scenario.set_init_single(
        'tl', init_trfficlight, (TLMode.GREEN,)
    )


    # ----------- Simulate single: Uncomment this block to perform single simulation -------------
    # trace = scenario.simulate(80, 0.1)
    # trace = scenario.verify(80, 0.1)
    # pp_fix(reach_at_fix(trace, 0, 79.91))
    # pp_fix(reach_at_fix(trace))
    # pp_old(reach_at(trace, 0, 79.91))
    # pp_old(reach_at(trace))
    # print('Do fixed points exist in the scenario:', fixed_points_fix(trace, 80))
    # avg_vel, unsafe_frac, unsafe_init = eval_velocity([trace])
    # fig = go.Figure()
    # fig = simulation_tree_3d(trace, fig,\
    #                           0,'time', 1,'x',2,'y')
    # fig.write_html('simulate.html', auto_open=True)

    ### testing out z3 set solver
    # P = RealVector('p', 4)
    # s = Solver()
    # or_chain = Or()
    # for i in range(len(P)):
    #     or_chain = Or(or_chain, P[i]**2<i)
    # s.add(P[0] > 5, Not(P[0]> 5)) ### sat inherently does AND due to needing to fulfill all conditions
    # print(s.sexpr())
    # print(s.check(), s.check()==unsat)
    ### testing helper functions
    # h1 = [np.array([0, 0, 0, 0]), np.array([0, 1, 1, 1])]
    # h2 = [np.array([0.5, -1, -0.5, -1.5]), np.array([0, 1, 2 , 3])]
    # print("Is h1 contained within h2:", contained_single(h1, h2))
    # print("Is h2 contained within h1:", contained_single(h2, h1))
    # print("Is h1 contained within h1:", contained_single(h1, h1))
    ###
    
    ### unit testing contain_all_fix
    # r1_1 = {0: [[0, 0], [1,0]]}
    # r1_2 = {1: [[0, 0], [1,0]], 2: [[0, 1], [1,1]]}
    # r2_1 = {0: [[-1, 0], [0.5, 0], [0.5, 0], [2, 0]]}
    # r3_1 = {0: [[-1, 0], [0.499, 0], [0.501, 0], [2, 0]]}
    # r3_2 = {0: [[-1, 0], [1, 0], [0, 1], [0.99, 2]]}
    # r4_1 = {0: [[-1, 0], [0.5, 0], [0.5, 1], [2, 1]]}
    # r4_2 = {0: [[-1, 0], [1, 0], [0, 1], [2, 1]]}
    # print(f'Is r1 contained within r2: {contain_all_fix(r1_1, r2_1)}')
    # print(f'Is r1 contained within r3: {contain_all_fix(r1_1, r3_1)}')
    # print(f'Is r1 contained within r4: {contain_all_fix(r1_1, r4_1)}')
    # print(f'Is r1_2 contained within r4_2: {contain_all_fix(r1_2, r4_2)}')
    # print(f'Is r1_2 contained within r4_2: {contain_all_fix(r3_2, r4_2)}')


    ###
    ball_scenario = BallScenario().scenario
    ball_scenario_branch = BallScenarioBranch().scenario

    # # ball_scenario_branch_nt = BallScenarioBranchNT().scenario ### this scenario's verify doesn't really make any sense given its simulate -- figure out why
    # # ## trying to verify with two agents in NT takes forever for some reason
    # # trace = ball_scenario_branch_nt.verify(80, 0.1) 
    # # trace = ball_scenario_branch_nt.simulate(80, 0.1)

    trace = ball_scenario_branch.verify(40, 0.1)
    sim = ball_scenario_branch.simulate(40, 0.1)
    # pp_fix(reach_at_fix(trace)) ### print out more elegantly, for example, line breaks and cut off array floats, test out more thoroughly
    # pp_fix(reach_at_fix(trace, 0, 39.91))
    print(fixed_points_fix(trace, 40, 0.01))

    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig = reachtube_tree_slice(trace, None, fig, 1, 2, [1, 2], "fill", "trace", plot_color=colors[1:])
    fig = simulation_tree(sim, None, fig, 1, 2, [1, 2], "fill", "trace", plot_color=colors[2:])
    fig.show()

    ###
    # -----------------------------------------

    # init_dict_list= sample_init(scenario, num_sample=50)
    # traces = scenario.simulate_multi(80, 0.1,
    #      init_dict_list=init_dict_list)
    # fig = go.Figure()
    # avg_vel, unsafe_frac, unsafe_init = eval_velocity(traces)
    # print(f"###### Average velocity {avg_vel}, Unsafe fraction {unsafe_frac}, Unsafe init {unsafe_init}")
    # for trace in traces:
    #     fig = simulation_tree_3d(trace, fig,\
    #                               0,'time', 1,'x',2,'y')
    # fig.show()
