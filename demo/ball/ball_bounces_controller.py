from enum import Enum, auto
import copy

class BallMode(Enum):
    # NOTE: Any model should have at least one mode
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)

class State:
    '''Defines the state variables of the model
        Both discrete and continuous variables
    '''
    x= 0.0
    y = 0.0
    vx = 0.0
    vy = 0.0
    mode: BallMode

    def __init__(self, x, y, vx, vy, ball_mode: BallMode):
        pass


def decisionLogic(ego: State):
    '''Computes the possible mode transitions'''
    output = copy.deepcopy(ego)
    '''TODO: Ego and output variable names should be flexible but 
    currently these are somehow harcoded with the sensor'''
    # Stores the prestate first
    if ego.x < 0:
        output.vx = -ego.vx
        output.x = 0
    if ego.y < 0:
        output.vy = -ego.vy
        output.y = 0
    if ego.x > 20:
        # TODO: Q. If I change this to ego.x >= 20 then the model does not work.
        # I suspect this is because the same transition can be take many, many times.
        # We need to figure out a clean solution
        output.vx = -ego.vx
        output.x = 20
    if ego.y > 20:
        output.vy = -ego.vy
        output.y = 20
    '''  if ego.x - others[1].x < 1 and ego.y - others[1].y < 1:
        output.vy = -ego.vy
        output.vx = -ego.vx'''
    # TODO: We would like to be able to write something like this, but currently not allowed.
    return output
