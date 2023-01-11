from verse.plotter.plotter2D import *
from robot_agent import RobotAgent
# from verse.map.example_map.simple_map2 import SimpleMap3
from verse import Scenario
from enum import Enum, auto
import copy

n = 5
flash = 2
alpha = 1.05

class RobotMode(Enum):
    Wait = auto()

class FLMode(Enum):
    One = auto()
    Zero = auto()

class State:
    x: float
    fl_mode: FLMode
    mode: RobotMode

    def __init__(self, x, fl_mode: FLMode, robot_mode: RobotMode):
        pass

def decisionLogic(ego: State, others:List[State]):
    output = copy.deepcopy(ego)
    if ego.x > flash:
        output.fl_mode = FLMode.One
        output.x = 0
    if ego.fl_mode == FLMode.One:
        output.fl_mode = FLMode.Zero
    if any(other.fl_mode == FLMode.One for other in others):
        output.x = alpha*ego.x
        if ego.x > flash/alpha:
            output.x = 0
    assert not all(other.x == ego.x for other in others), "Sync"
    return output

if __name__ == "__main__":
    swarm = Scenario()
    swarm_controller = './robot_swarm/robot_swarm.py'
    for i in range(n):
        myagent = RobotAgent(f'robot_{i+1}', file_name=swarm_controller)
        swarm.add_agent(myagent)
    # tmp_map = SimpleMap3()
    # swarm.set_map(tmp_map)
    init = []
    init_mode = []
    for i in range(n):
        init_l = [float(i/n)]
        init_u = [float(i/n)]
        init.append([init_l,init_u])
        init_mode.append((FLMode.Zero, RobotMode.Wait,))
    swarm.set_init(init,init_mode)
    swarm.config.init_seg_length = 1
    traces = swarm.verify(10, 0.01)
    fig = go.Figure()
    fig = reachtube_tree(
        traces, None, fig, 0, 1, [0, 1], 'lines', 'trace')
    fig.show()
