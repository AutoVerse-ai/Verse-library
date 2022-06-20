from enum import Enum, auto
import copy
from typing import List
# from dryvr_plus_plus.scene_verifier.map.lane import Lane

class BallMode(Enum):
    # Any model should have at least one mode
    Normal = auto()
    # The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)

class LaneMode(Enum):
    Lane0 = auto()
    # For now this is a dummy notion of Lane

class State:
    '''Defines the state variables of the model
        Both discrete and continuous variables
    '''
    x:float
    y = 0.0
    vx = 0.0
    vy = 0.0
    mode: BallMode
    lane_mode: LaneMode
    def __init__(self, x, y, vx, vy, ball_mode:BallMode, lane_mode:LaneMode):
        pass

def controller(ego:State, others:State):
    '''Computes the possible mode transitions'''
    output = copy.deepcopy(this)
    '''Ego and output variable names should be flexible but 
    currently these are somehow harcoded with the sensor'''
    # Stores the prestate first
    if ego.x<0:
        output.vx = -ego.vx
        output.x=0
    if ego.y<0:
        output.vy = -ego.vy
        output.y=0
    if ego.x>20:
        # Q. If I change this to ego.x >= 20 then the model does not work.
        # I suspect this is because the same transition can be take many, many times.
        # We need to figure out a clean solution
        output.vx = -ego.vx
        output.x=20
    if ego.y>20:
        output.vy = -ego.vy
        output.y=20
  '''  if ego.x - others[1].x < 1 and ego.y - others[1].y < 1:
        output.vy = -ego.vy
        output.vx = -ego.vx'''
  # We would like to be able to write something like this, but currently not allowed. 
    return output



from dryvr_plus_plus.example.example_agent.ball_agent import BallAgent
from dryvr_plus_plus.scene_verifier.scenario.scenario import Scenario
from dryvr_plus_plus.example.example_map.simple_map2 import SimpleMap3
from dryvr_plus_plus.example.example_sensor.fake_sensor import FakeSensor4
from dryvr_plus_plus.plotter.plotter2D import *
import plotly.graph_objects as go

if __name__ == "__main__":
    ball_controller = '/Users/mitras/Dpp/GraphGeneration/demo/ball_bounces.py'
    bouncingBall = Scenario()
    myball1 = BallAgent('red-ball', file_name=ball_controller)
    myball2 = BallAgent('green-ball', file_name=ball_controller)
    bouncingBall.add_agent(myball1)
    bouncingBall.add_agent(myball2)
    #
    tmp_map = SimpleMap3()
    bouncingBall.set_map(tmp_map)
    # If there is no useful map, then we should not have to write this.
    # Default to some empty map
    bouncingBall.set_sensor(FakeSensor4())
    # There should be a way to default to some well-defined sensor
    # for any model, without having to write an explicit sensor
    bouncingBall.set_init(
        [
            [[5, 10, 2, 2], [5, 10, 2, 2]],
            [[15, 1, 1, -2], [15, 1, 1, -2]]
        ],
        [
            (BallMode.Normal, LaneMode.Lane0),
            (BallMode.Normal, LaneMode.Lane0)
        ]
    )
    # WE should be able to initialize each of the balls separately
    # this may be the cause for the VisibleDeprecationWarning
    # Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
    # "-2 \leq myball1.x + myball2.x \leq 5"
    traces = bouncingBall.simulate(20)
    # There should be a print({traces}) function
    fig = go.Figure()
    fig = plotly_simulation_anime(traces, tmp_map, fig)
    fig.show()

