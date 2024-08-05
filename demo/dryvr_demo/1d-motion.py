from verse.plotter.plotter2D import *
from line_agent import LineAgent
from verse.map.example_map.simple_map2 import SimpleMap3
from verse import Scenario, ScenarioConfig
from enum import Enum, auto
import copy

# from verse.map import Lane

class PointMode(Enum):
    '''STOPPED: velocity =0, LEFT: -1, RIGHT: +1'''
    STOPPED = auto()
    LEFT = auto()
    RIGHT = auto()

class State:
    '''Defines the state variables of the model '''
    x: float
    vx = 0.0
    mode: PointMode

    def __init__(self, x, vx, ball_mode: PointMode):
        pass

def decisionLogic(ego: State):
    '''Computes the possible mode transitions'''
    # Stores the prestate first and updates the poststate
    # Variables that are not assigned are assumed to be unchanged
    cr = 0.85
    output = copy.deepcopy(ego)
    if ego.mode == PointMode.STOPPED and ego.vx !=0:
        output.vx = 0
    if ego.mode == PointMode.LEFT and ego.vx !=-1:
        output.vx = -1
    if ego.mode == PointMode.RIGHT and ego.vx !=1:
        output.vx = 1

    # bouncing type logic from before
    # if ego.x < 0:
    #     output.vx = -ego.vx*cr
    #     output.x = 0
    # if ego.x > 20:
    #     output.vx = -ego.vx*cr
    #     output.x = 20
    return output


if __name__ == "__main__":
    pointPair = Scenario(ScenarioConfig(parallel=False))  # scenario too small, parallel too slow
    CONTROLLER = "./demo/dryvr_demo/1d-motion.py"
    pt1 = LineAgent("red-ball", file_name=CONTROLLER)
    pt2 = LineAgent("green-ball", file_name=CONTROLLER)
    pointPair.add_agent(pt1)
    pointPair.add_agent(pt2)
    pointPair.set_init(
        [[[5, 1], [6, 1]], [[2, 1], [3, 1]]],
        [(PointMode.RIGHT,), (PointMode.LEFT,)],
    )
    fig = go.Figure()

    # Simulation code
    # traces = pointPair.simulate_simple(10, 0.01, 6)
    # Input arguments for simulation_tree
    # trace to be plotted, ?, fig to add to, x-axis data, y-axis data, ?, style
    # fig = simulation_tree(traces, None, fig, 0, 1, [0, 1], "fill", "trace")
    # Verification code
    traces = pointPair.verify(10, 0.1, params={"bloating_method": "GLOBAL"}) 
    # traces = pointPair.verify(10, 0.1) 
    fig = reachtube_tree(traces, None, fig, 0, 1, [0, 1], "lines", "trace")
       

    # path 1
    # added decision logic with different modes 1,0,-1
    # check if mode / agent_mode is the right way to model 
    # check work process for where new type of agents should be added
    # add sensor model
    # add two different control inputs +1, -1
    # write reachability algorithm based on partitioning initial sets
    # this will have to use joint reachability, based on containment in distinguishable sets. 
    # path 2
    # add nonlinear drift
    fig.show()