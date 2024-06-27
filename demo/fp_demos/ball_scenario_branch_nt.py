from verse.plotter.plotter2D import *
from verse.agents.example_agent.ball_agent import BallAgent
from verse.map.example_map.simple_map2 import SimpleMap3
from verse import Scenario, ScenarioConfig
from enum import Enum, auto
import copy
import os 

# from verse.map import Lane

class BallMode(Enum):
    '''NOTE: Any model should have at least one mode
    The one mode of this automation is called "Normal" and auto assigns it an integer value.'''
    INTER = auto()
    REVERSE = auto()
    NORMAL = auto()

class State:
    '''Defines the state variables of the model
    Both discrete and continuous variables
    '''
    x: float
    y = 0.0
    vx = 0.0
    vy = 0.0
    mode: BallMode

    def __init__(self, x, y, vx, vy, ball_mode: BallMode):
        pass

def decisionLogic(ego: State):
    '''Computes the possible mode transitions'''
    # Stores the prestate first
    output = copy.deepcopy(ego)
    if ego.mode == BallMode.INTER:
        output.mode = BallMode.NORMAL
    if ego.mode == BallMode.INTER:
        output.mode = BallMode.REVERSE
    if ego.x < 0:
        output.vx = -ego.vx
        output.x = 0
    if ego.y < 0:
        output.vy = -ego.vy
        output.y = 0
    if ego.x > 20:
        output.vx = -ego.vx
        output.x = 20
        # if ego.mode==BallMode.REVERSE:
        #     output.y = 0
        # output.mode = BallMode.INTER
    if ego.y > 20:
        output.vy = -ego.vy
        output.y = 20
        # if ego.mode==BallMode.REVERSE:
        #     output.x = 0
        # output.mode = BallMode.INTER
    return output

class BallScenarioBranchNT:
    scenario: Scenario

    def __init__(self) -> None:
        self.scenario = Scenario(ScenarioConfig(parallel=False))
        script_dir = os.path.realpath(os.path.dirname(__file__))
        BALL_CONTROLLER = os.path.join(script_dir, "ball_scenario_branch_nt.py")
        myball1 = BallAgent("red-ball", file_name=BALL_CONTROLLER)
        myball2 = BallAgent("green-ball", file_name=BALL_CONTROLLER)
        self.scenario.add_agent(myball1)
        self.scenario.add_agent(myball2)
        self.scenario.set_init(
            [[[0, 0, 2, 2], [0, 0, 2, 2]], 
             [[0, 0, 2, 2], [0, 0, 2, 2]]
             ], # modified from original to check for fixed points, see below
            # [[[5, 10, 2, 2], [5, 10, 2, 2]],],
            [(BallMode.INTER,)
             , (BallMode.INTER,)
             ],
        )

if __name__ == "__main__":
    #Defining and using a scenario involves the following 5 easy steps:
    #1. creating a basic scenario object with Scenario()
    #2. defining the agents that will populate the object, here we have two ball agents
    #3. adding the agents to the scenario using .add_agent()
    #4. initializing the agents for this scenario.
    #   Note that agents are only initialized *in* a scenario, not individually outside a scenario
    #5. genetating the simulation traces or computing the reachable states
    bouncingBall = Scenario(ScenarioConfig(parallel=False))  # scenario too small, parallel too slow
    BALL_CONTROLLER = "./demo/ball/ball_bounces.py"
    myball1 = BallAgent("red-ball", file_name=BALL_CONTROLLER)
    myball2 = BallAgent("green-ball", file_name=BALL_CONTROLLER)
    bouncingBall.add_agent(myball1)
    bouncingBall.add_agent(myball2)
    bouncingBall.set_init(
        [[[5, 10, 2, 2], [5, 10, 2, 2]], [[15, 1, 1, -2], [15, 1, 1, -2]]],
        [(BallMode.NORMAL,), (BallMode.NORMAL,)],
    )
    # TODO: We should be able to initialize each of the balls separately
    # this may be the cause for the VisibleDeprecationWarning
    # TODO: Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
    # "-2 \leq myball1.x + myball2.x \leq 5"
    traces = bouncingBall.simulate_simple(40, 0.01, 6)
    # TODO: There should be a print({traces}) function
    fig = go.Figure()
    fig = simulation_tree(traces, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig.show()