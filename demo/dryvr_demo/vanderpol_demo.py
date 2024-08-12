from origin_agent import vanderpol_agent
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto

import time

from verse.analysis import ReachabilityMethod
from verse.stars.starset import StarSet
from verse.sensor.base_sensor_stars import *
import polytope as pc

from new_files.star_diams import *


class AgentMode(Enum):
    Default = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/vanderpol_controller.py'
    scenario = Scenario(ScenarioConfig(parallel=False))

    car = vanderpol_agent('car1', file_name=input_code_name)
    # scenario.add_agent(car)
    # # car = vanderpol_agent('car2', file_name=input_code_name)
    # # scenario.add_agent(car)
    # # scenario.set_sensor(FakeSensor2())
    # # modify mode list input
    # scenario.set_init(
    #     [
    #         [[1.25, 2.25], [1.25, 2.25]],
    #         # [[1.55, 2.35], [1.55, 2.35]]
    #     ],
    #     [
    #         tuple([AgentMode.Default]),
    #         # tuple([AgentMode.Default]),
    #     ]
    # )

    initial_set_polytope = pc.box2poly([[1.25,1.26],[2.25,2.26]])
    car.set_initial(StarSet.from_polytope(initial_set_polytope), (AgentMode.Default,))

    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.add_agent(car)

    scenario.set_sensor(BaseStarSensor())

    start_time = time.time()
    traces = scenario.verify(7, 0.05)
    run_time = time.time() - start_time
    # fig = go.Figure()

    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
    #                       'lines', 'trace')
    # fig.show()

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    plot_reachtube_stars(traces, None, 1, 2,1)

    print("time")
    print(run_time)
    print("diams")
    diams = time_step_diameter(traces, 7, 0.05)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])
