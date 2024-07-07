from origin_agent import vanderpol_agent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *

from verse.stars.starset import *
import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()



if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/vanderpol_controller.py"
    scenario = Scenario(ScenarioConfig(parallel=False,reachability_method=ReachabilityMethod.STAR_SETS))

    car = vanderpol_agent("car1", file_name=input_code_name)
    scenario.add_agent(car)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    basis = np.array([[1, 0], [0, 1]])
    center = np.array([3,3])
    C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
    g = np.array([1,1,1,1])

    ### how do I instantiate a scenario with a starset instead of a hyperrectangle?

    scenario.set_init(
        [
            # [[1.25, 2.25], [1.25, 2.25]],
            # [[1.55, 2.35], [1.55, 2.35]]
            [StarSet(center, basis, C, g)]
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ],
    )
    traces = scenario.verify(7, 0.05)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2], "lines", "trace")
    fig.show()
