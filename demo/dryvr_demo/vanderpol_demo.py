from origin_agent import vanderpol_agent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *

from verse.stars.starset import *
import plotly.graph_objects as go
from enum import Enum, auto

from verse.sensor.base_sensor_stars import *

class AgentMode(Enum):
    Default = auto()

def plot_stars(stars: List[StarSet], dim1: int = None, dim2: int = None):
    for star in stars:
        x, y = np.array(star.get_verts(dim1, dim2))
        plt.plot(x, y, lw = 1)
        centerx, centery = star.get_center_pt(0, 1)
        plt.plot(centerx, centery, 'o')
    plt.show()


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/vanderpol_controller.py"
    scenario = Scenario(ScenarioConfig(parallel=False))

    car = vanderpol_agent("car1", file_name=input_code_name)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    basis = np.array([[1, 0], [0, 1]]) * np.diag([0.1, 0.1])
    center = np.array([1.40,2.30])
    C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
    g = np.array([1,1,1,1])

    ### how do I instantiate a scenario with a starset instead of a hyperrectangle?

    car.set_initial(
            # [[1.25, 2.25], [1.25, 2.25]],
            # [[1.55, 2.35], [1.55, 2.35]]
            StarSet(center, basis, C, g)
        ,
            tuple([AgentMode.Default])
            # tuple([AgentMode.Default]),
        ,
    )

    scenario.add_agent(car)
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.set_sensor(BaseStarSensor())
    traces = scenario.verify(7, 0.05)
    
    car1 = traces.nodes[0].trace['car1']
    car1 = [star[1] for star in car1]
    
    for star in car1:
        print(star.center, star.basis, star.C, star.g, '\n --------')
    plot_stars(car1, 0, 1)
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 1, [0, 1], "lines", "trace")
    # fig.show()
