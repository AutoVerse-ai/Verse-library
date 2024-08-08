from origin_agent import thermo_agent
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.thermo_sensor import ThermoSensor
import plotly.graph_objects as go
from enum import Enum, auto
from verse.analysis.verifier import ReachabilityMethod

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
    # scenario.add_agent(car)
    # scenario.set_sensor(ThermoSensor())
    # modify mode list input
    # scenario.set_init(
    #     [
    #         [[75.0, 0.0, 0.0], [76, 0.0, 0.0]],
    #     ],
    #     [
    #         tuple([ThermoMode.ON]),
    #     ],
    # )
    # traces = scenario.simulate(3.5, 0.05)
    # fig = go.Figure()
    # fig = simulation_tree(traces, None, fig, 2, 1, [2, 1], "lines", "trace")

    # basis = np.array([[ 0.34474,-0.00154],[-0.00012,-0.02748]])
    # center = np.array([-960.47483,6.7487])
    # C = np.transpose(np.array([[1, 0, -1, 0], [0, 1, 0, -1]]))
    # g = np.array([3030.84222,39.5072,-2989.43095,-35.91564,])
    # StarSet(center, basis, C, g).plot()
    # exit()
    ### issue could be twofold as basestarsensor is also replace the bespoke therm sensor, which could be causing the errors
    ### big issue: why are timers unsynched/not a multple of the ts? e.g., seeing this average time: [82.94445  1.15482  0.1    ]?
    ### deepest issue though is why would post_cont_pca ever unsat for this?
    basis = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) * np.diag([0.1, 0, 0]) # this doesn't actually make sense, but not sure how algorithm actually handles 1d polytopes
    center = np.array([75.5,0,0])
    C = np.transpose(np.array([[1,-1,0,0,0,0],[0,0,1,-1,0,0], [0,0,0,0,1,-1]]))
    g = np.array([1,1,1,1,1,1])

    ### how do I instantiate a scenario with a starset instead of a hyperrectangle?

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
    scenario.set_sensor(BaseStarSensor())

    scenario.verify(2, 0.1)
    # fig.show()
