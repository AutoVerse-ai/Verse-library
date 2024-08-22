from verse.plotter.plotter2D import *
from verse.agents.example_agent.ball_agent import BallAgent
from verse.map.example_map.simple_map2 import SimpleMap3
from verse import Scenario, ScenarioConfig
from enum import Enum, auto
import copy
import os 
from verse.scenario.scenario import ReachabilityMethod

from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *

class BallMode(Enum):
    Normal = auto()

def ball_bounce_test():
    bouncingBall = Scenario(ScenarioConfig(parallel=False, reachability_method=ReachabilityMethod.STAR_SETS))  # scenario too small, parallel too slow
    script_dir = os.path.realpath(os.path.dirname(__file__))
    ball_controller = os.path.join(script_dir, './test_controller/ball_controller.py')
    ball_controller2 = os.path.join(script_dir, './test_controller/ball_controller2.py')
    myball1 = BallAgent("red-ball", file_name=ball_controller)
    myball2 = BallAgent("green-ball", file_name=ball_controller2)

    basis = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) * np.diag([0.001, 0.001, 0.001, 0.001])  
    center = np.array([5, 10, 2, 2])
    center2 = np.array([15, 1, 1, -2])
    C = np.transpose(np.array([[1,-1,0,0, 0, 0, 0, 0],[0,0,1,-1, 0, 0, 0,0], [0,0,0,0,1,-1, 0, 0],[0,0,0,0,0,0,1,-1]]))
    g = np.array([1,1,1,1,1,1,1,1])

    myball1.set_initial(
        StarSet(center, basis, C, g),
        tuple([BallMode.Normal])
    )

    myball2.set_initial(
        StarSet(center2, basis, C, g),
        tuple([BallMode.Normal])
    )

    bouncingBall.add_agent(myball1)
    bouncingBall.add_agent(myball2)
    # bouncingBall.set_init(
    #     [[[5, 10, 2, 2], [5, 10, 2, 2]], [[15, 1, 1, -2], [15, 1, 1, -2]]],
    #     [(BallMode.Normal,), (BallMode.Normal,)],
    # )
    # TODO: WE should be able to initialize each of the balls separately
    # this may be the cause for the VisibleDeprecationWarning
    # TODO: Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
    # "-2 \leq myball1.x + myball2.x \leq 5"
    #traces = bouncingBall.simulate(40, 0.01, 10)

    bouncingBall.set_sensor(BaseStarSensor())

    ### why do we keep having nodes that start at 0.5? occurs when initial set is not a single point i.e., basis not the zero matrix
    ### for some reason this also doesn't work for dryvr either when initial set is not a single point:    assert np.all(df >= 0) AssertionError
    traces = bouncingBall.verify(10, 0.1)
    # TODO: There should be a print({traces}) function
    # fig = go.Figure()
    # fig = simulation_tree(traces, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig = None

    return traces, fig    

if __name__ == "__main__":
    _, fig = ball_bounce_test()

    # fig.show()
