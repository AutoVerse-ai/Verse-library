from origin_agent_spacecraft import spacecraft_linear_agent, spacecraft_linear_agent_nd
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor
import time
import plotly.graph_objects as go
from enum import Enum, auto
#starset
from verse.analysis import ReachabilityMethod
from verse.stars.starset import StarSet
from verse.sensor.base_sensor_stars import *
import polytope as pc


class CraftMode(Enum):
    Approaching = auto()
    Rendezvous = auto()
    Aborting = auto()


if __name__ == "__main__":
    #ZERO
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controller.py'
    scenario = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))#ScenarioConfig(parallel=False))

    car = spacecraft_linear_agent('test', file_name=input_code_name)

    #post starset
    initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    car.set_initial(StarSet.from_polytope(initial_set_polytope), (CraftMode.Approaching,))

    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.add_agent(car)

    scenario.set_sensor(BaseStarSensor())


    start_time = time.time()

    traces = scenario.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRA01",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })

    

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    plot_reachtube_stars(traces, None, 1, 2,10)