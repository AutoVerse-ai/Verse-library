from enum import Enum, auto
import copy

class BallMode(Enum):
    Normal = auto()

class State:
    x: float
    y = 0.0
    vx = 0.0
    vy = 0.0
    mode: BallMode

    def __init__(self, x, y, vx, vy, ball_mode: BallMode):
        pass

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.x < 0:
        output.vx = -ego.vx
        output.x = 0
    if ego.y < 0:
        output.vy = -ego.vy
        output.y = 0
    if ego.x > 20:
        output.vx = -ego.vx
        output.x = 20
    if ego.y > 20:
        output.vy = -ego.vy
        output.y = 20
    return output