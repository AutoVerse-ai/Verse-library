from origin_agent import thermo_agent
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.thermo_sensor import ThermoSensor
import plotly.graph_objects as go
from enum import Enum, auto
from verse.analysis.verifier import ReachabilityMethod
from z3 import *
from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *

class ThermoMode(Enum):
    ON = auto()
    OFF = auto()


### tests 
if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/thermo_controller.py"
    scenario = Scenario(ScenarioConfig(parallel=False))

    car = thermo_agent("test", file_name=input_code_name)

    basis = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) * np.diag([0.1, 0, 0]) 
    center = np.array([75.5,0,0])
    C = np.transpose(np.array([[1,-1,0,0,0,0],[0,0,1,-1,0,0], [0,0,0,0,1,-1]]))
    g = np.array([1,1,1,1,1,1])

    car.set_initial(
            # [[1.25, 2.25], [1.25, 2.25]],
            # [[1.55, 2.35], [1.55, 2.35]]
            StarSet(center, basis, C, g)
        ,
            tuple([ThermoMode.ON])
            # tuple([AgentMode.Default]),
        ,
    )

    scenario.add_agent(car)
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    # scenario.config.pca = False
    scenario.set_sensor(BaseStarSensor())

    trace = scenario.verify(7, 0.1)
    plot_reachtube_stars(trace)
    # fig.show()
