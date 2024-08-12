from origin_agent import gearbox_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.scenario import ScenarioConfig
import plotly.graph_objects as go
from enum import Enum, auto
import time

#starset
from verse.analysis import ReachabilityMethod
from verse.stars.starset import StarSet
import polytope as pc
from verse.sensor.base_sensor_stars import *

from new_files.star_diams import *

class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


if __name__ == "__main__":
    #ONE
    input_code_name = './demo/dryvr_demo/sleeve_controller.py'
    config=ScenarioConfig(init_seg_length=1, parallel=False)
    scenario = Scenario(config=config)

    car = gearbox_agent('sleeve', file_name=input_code_name)
    #scenario.add_agent(car)

    #pre starset
    '''
    scenario.set_init(
        [
            [[-0.0168, 0.0029, 0, 0, 0,0,1], [-0.0166, 0.0031, 0, 0, 0,0,1]],
        ],
        [
            tuple([AgentMode.Free]),
        ]
    )
    '''
    initial_set_polytope = pc.box2poly([[-0.0168, -0.0166], [0.0029, 0.0031], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1]])

    car.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([AgentMode.Free]))
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.add_agent(car)

    scenario.set_sensor(BaseStarSensor())

    start_time = time.time()
    traces = scenario.verify(.21, 1e-4)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Gear",
        "setup": "GRBX01",
        "result": "0",
        "time": run_time,
        "metric2": 'n/a',
        "metric3": "n/a",
    })

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    #over_approx_rectangle(traces, 0.04, 1e-4)

    plot_reachtube_stars(traces, None, 1 , 2, 5)

    # diams = time_step_diameter(traces, .11, 1e-4)
    # print(len(diams))
    # print(sum(diams))
    # print(diams[0])
    # print(diams[-1])


    #traces.dump('./demo/gearbox/output.json')
    #pre starset
    '''
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], 'lines', 'trace', sample_rate=1)
    #fig.show()
    
    '''



    
    input_code_name = './demo/dryvr_demo/sleeve_controller.py'
    config = ScenarioConfig(init_seg_length=1, parallel=False)
    scenario = Scenario(config=config)

    car = gearbox_agent('sleeve', file_name=input_code_name)
    #scenario.add_agent(car)

    #pre starset

    '''
    scenario.set_init(
        [
            [[-0.01675,0.00285 , 0, 0, 0, 0, 1], [-0.01665,0.00315, 0, 0, 0, 0, 1]],
        ],
        [
            tuple([AgentMode.Free]),
        ]
    )
    '''

    initial_set_polytope = pc.box2poly([[-0.01675, -0.01665], [0.00285, 0.00315], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1]])

    car.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([AgentMode.Free]))
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.add_agent(car)
    
    scenario.set_sensor(BaseStarSensor())

    start_time = time.time()
    traces = scenario.verify(.11, 1e-4)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Gear",
        "setup": "GRBX02",
        "result": "0",
        "time": run_time,
        "metric2": 'n/a',
        "metric3": "n/a",
    })

    diams = time_step_diameter(traces, .11, 1e-4)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])

    
    #traces.dump('./demo/gearbox/output.json')
    #pre starset
    '''

    '''
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], 'lines', 'trace', sample_rate=1)
    #fig.show()
    
