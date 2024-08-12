from origin_agent import robertson_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto
import time

from verse.analysis import ReachabilityMethod
from verse.stars.starset import StarSet
from verse.sensor.base_sensor_stars import *
import polytope as pc


class AgentMode(Enum):
    Default = auto()

input_code_name = './demo/dryvr_demo/robertson_controller.py'

scenario1 = Scenario(ScenarioConfig(parallel=False))

car1 = robertson_agent('car1', file_name=input_code_name, beta = 1e3, gamma=1e7)
# scenario1.add_agent(car1)
# scenario1.set_init(
#     [
#         [[1, 0, 0], [1, 0, 0]],
#     ],
#     [
#         tuple([AgentMode.Default]),
#         # tuple([AgentMode.Default]),
#     ]
# )

initial_set_polytope = pc.box2poly([[1,1.000001],[0,0.000001],[0,0.000001]])
car1.set_initial(StarSet.from_polytope(initial_set_polytope), (AgentMode.Default,))

scenario1.config.reachability_method = ReachabilityMethod.STAR_SETS
scenario1.add_agent(car1)

scenario1.set_sensor(BaseStarSensor())

start_time = time.time()

traces1 = scenario1.verify(40, .1)
# traces1 = compute_xyz(traces1)
run_time = time.time() - start_time

print({
    "tool": "verse",
    "benchmark": "ROBE21",
    "setup": "B3G7",
    "result": "1",
    "time": run_time,
    "metric2": 400,
    # "metric3": traces1.nodes[0].trace['car1'][-1][1] + traces1.nodes[0].trace['car1'][-1][2] +
    #            traces1.nodes[0].trace['car1'][-1][3] - (
    #                    traces1.nodes[0].trace['car1'][-2][1] + traces1.nodes[0].trace['car1'][-2][2] +
    #                    traces1.nodes[0].trace['car1'][-2][3]),
})
# fig = reachtube_tree(traces1, None, fig, 1, 2, [1, 2], 'lines', 'trace')
# fig.show()

import plotly.graph_objects as go
from verse.plotter.plotterStar import *

plot_reachtube_stars(traces1, None, 1, 2,1)