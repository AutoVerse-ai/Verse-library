from verse.plotter.plotter2D import *
from verse.agents.example_agent.robot_agent import RobotAgent
from verse import Scenario
from enum import Enum, auto
import copy

n = 5
flash = 2
alpha = 1.05


class RobotMode(Enum):
    Wait = auto()


class State:
    x: float
    mode: RobotMode

    def __init__(self, x, robot_mode: RobotMode):
        pass


def decisionLogic(ego: State, others: List[State]):
    output = copy.deepcopy(ego)
    if ego.x > flash:
        output.x = 0
    if any(other.x >= flash for other in others):
        output.x = alpha * ego.x
        if ego.x >= flash / alpha:
            output.x = 0
    assert not all(other.x == ego.x for other in others), "Sync"
    return output


if __name__ == "__main__":
    swarm = Scenario()
    swarm_controller = "./demo/robot_swarm/lsync.py"
    for i in range(n):
        myagent = RobotAgent(f"robot_{i+1}", file_name=swarm_controller)
        swarm.add_agent(myagent)
    init = []
    init_mode = []
    for i in range(n):
        init_l = [float(i / n)]
        init_u = [float(i / n)]
        init.append([init_l, init_u])
        init_mode.append((RobotMode.Wait,))
    swarm.set_init(init, init_mode)
    traces = swarm.simulate_simple(10, 0.01)
    traces.visualize()
    # fig = go.Figure()
    # fig = simulation_tree(
    #    traces, None, fig, 0, 1, [0, 1], 'lines', 'trace')
    # fig.show()
