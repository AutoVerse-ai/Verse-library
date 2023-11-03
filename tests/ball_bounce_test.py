from tests.test_controller import ball_controller2
from verse.plotter.plotter2D import *
from verse.agents.example_agent.ball_agent import BallAgent
from verse.map.example_map.simple_map2 import SimpleMap3
from verse import Scenario, ScenarioConfig
from enum import Enum, auto
import copy
import os 

class BallMode(Enum):
    Normal = auto()

def ball_bounce_test():
    bouncingBall = Scenario(ScenarioConfig(parallel=True, print_level=1))  # scenario too small, parallel too slow
    script_dir = os.path.realpath(os.path.dirname(__file__))
    ball_controller = os.path.join(script_dir, './test_controller/ball_controller.py')
    ball_controller2 = os.path.join(script_dir, './test_controller/ball_controller2.py')
    myball1 = BallAgent("red-ball", file_name=ball_controller)
    myball2 = BallAgent("green-ball", file_name=ball_controller2)
    bouncingBall.add_agent(myball1)
    bouncingBall.add_agent(myball2)
    bouncingBall.set_init(
        [[[5, 10, 2, 2], [5, 10, 2, 2]], [[15, 1, 1, -2], [15, 1, 1, -2]]],
        [(BallMode.Normal,), (BallMode.Normal,)],
    )
    # TODO: WE should be able to initialize each of the balls separately
    # this may be the cause for the VisibleDeprecationWarning
    # TODO: Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
    # "-2 \leq myball1.x + myball2.x \leq 5"
    #traces = bouncingBall.simulate(40, 0.01, 10)
    traces = bouncingBall.verify(20, 0.01, 10)
    # TODO: There should be a print({traces}) function
    fig = go.Figure()
    fig = simulation_tree(traces, None, fig, 1, 2, [1, 2], "fill", "trace")

    return traces, fig    

if __name__ == "__main__":
    _, fig = ball_bounce_test()
    fig.show()
