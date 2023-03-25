from verse.plotter.plotter2D import *
from verse.agents.example_agent.ball_agent import BallAgent
from verse.map.example_map.simple_map2 import SimpleMap3
from verse import Scenario
from enum import Enum, auto
import copy


# from verse.map import Lane


class BallMode(Enum):
    # NOTE: Any model should have at least one mode
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)

# class TrackMode(Enum):
#     Lane0 = auto()
#     #For now this is a dummy notion of Lane






if __name__ == "__main__":
    ''' Defining and using a  scenario involves the following 5 easy steps:
        1. creating a basic scenario object with Scenario()
        2. defining the agents that will populate the object, here we have two ball agents
        3. adding the agents to the scenario using .add_agent()
        4. initializing the agents for this scenario. 
            Note that agents are only initialized *in* a scenario, not individually outside a scenario
        5. genetating the simulation traces or computing the reachable states    
    '''
    bouncingBall = Scenario()
    ball_controller = './demo/ball/ball_bounces_controller.py'
    myball1 = BallAgent('red-ball', file_name=ball_controller)
    #myball2 = BallAgent('green-ball', file_name=ball_controller)
    bouncingBall.add_agent(myball1)
    #bouncingBall.add_agent(myball2)
    bouncingBall.set_init(
        [
            [[5, 10, 2, 2], [10, 15, 2, 2]]
            #[[15, 1, 1, -2], [15, 1, 1, -2]]
        ],
        [
            (BallMode.Normal,)
            #(BallMode.Normal,)
        ]
    )
    # TODO: WE should be able to initialize each of the balls separately
    # this may be the cause for the VisibleDeprecationWarning
    # TODO: Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
    # "-2 \leq myball1.x + myball2.x \leq 5"
    traces = bouncingBall.verify(40, .01 )
    # TODO: There should be a print({traces}) function
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2], 'lines', 'trace', combine_rect=3)

    # fig = simulation_tree(
    #     traces, None, fig, 1, 2, [1, 2], 'lines', 'trace')
    fig.show()
