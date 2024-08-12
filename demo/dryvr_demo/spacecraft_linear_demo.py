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

from new_files.star_diams import *

# import builtins
# from inspect import getframeinfo, stack
# original_print = print

# def print_wrap(*args, **kwargs):
#     caller = getframeinfo(stack()[1][0])
#     original_print("FN:",caller.filename,"Line:", caller.lineno,"Func:", caller.function,":::", *args, **kwargs)

# builtins.print = print_wrap


class CraftMode(Enum):
    Approaching = auto()
    Rendezvous = auto()
    Aborting = auto()


if __name__ == "__main__":
    #ZERO
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller.py'
    scenario = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))#ScenarioConfig(parallel=False))

    car = spacecraft_linear_agent('test', file_name=input_code_name)
    #scenario.add_agent(car)


    #pre starset
    '''
    # modify mode list input

    scenario.set_init(
        [
            [[-925, -425, 0, 0,0,0], [-875, -375, 0, 0, 0,0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    # traces = scenario.simulate(200, 1)
    # fig = go.Figure()
    # fig = simulation_anime(traces, None, fig, 1, 2, [
    #                        1, 2], 'lines', 'trace', sample_rate=1)
    # fig.show()
    '''
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

    #over_approx_rectangle(traces, 300, .1)

    

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    #plot_reachtube_stars(traces, None, 1, 2,10)


    diams = time_step_diameter(traces, 300, .1)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])


    #pre starset
    '''
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')
    fig.update_layout(
        xaxis_title="x", yaxis_title="y"
    )
    '''

    #fig.show()

    #ONE
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_na.py'
    scenario1 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car1 = spacecraft_linear_agent('test', file_name=input_code_name)
    #scenario1.add_agent(car1)


    #pre starset
    '''
    # modify mode list input

    scenario1.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )
    '''
    #post starset
    initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    car1.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    scenario1.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario1.add_agent(car1)

    scenario1.set_sensor(BaseStarSensor())

    start_time = time.time()

    traces = scenario1.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRNA01",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })

    #plot_reachtube_stars(traces, None, 1 , 2, 10)

    diams = time_step_diameter(traces, 300, .1)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])

    #plot_reachtube_stars(traces, None, 1, 2,1)

    #pre starset
    '''
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')
    fig.update_layout(
        xaxis_title="x", yaxis_title="y"
    )
    '''
    #fig.show()


    #TWO
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_02.py'
    scenario2 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car2 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    #scenario2.add_agent(car2)

    #pre starset
    '''
    # modify mode list input

    scenario2.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )
    '''

    #post starset
    initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    car2.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    scenario2.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario2.add_agent(car2)

    scenario2.set_sensor(BaseStarSensor())

    start_time = time.time()

    traces = scenario2.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRA02",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })

    #plot_reachtube_stars(traces, None, 1 , 2, 10)

    diams = time_step_diameter(traces, 300, .1)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])


    #pre starset
    '''
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    #fig.show()
    '''

    #THREE
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_03.py'
    scenario3 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car3 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    #scenario3.add_agent(car3)

    #pre starset
    '''
    # modify mode list input

    scenario3.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )
    '''

    #post starset
    initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    car3.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    scenario3.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario3.add_agent(car3)

    scenario3.set_sensor(BaseStarSensor())

    start_time = time.time()

    traces = scenario3.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRA03",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })

    #plot_reachtube_stars(traces, None, 1 , 2)

    diams = time_step_diameter(traces, 300, .1)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    #plot_reachtube_stars(traces, None, 1, 2,10)

    #pre starset
    '''
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')
    '''

    #fig.show()

    #FOUR
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_04.py'
    scenario4 = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    car4 = spacecraft_linear_agent('test', file_name=input_code_name)
    #scenario4.add_agent(car4)

    #pre starset
    '''
    # modify mode list input

    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )
    '''

    #post starset
    initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    car4.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    scenario4.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario4.add_agent(car4)

    scenario4.set_sensor(BaseStarSensor())

    start_time = time.time()

    traces = scenario4.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRA04",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })

    #plot_reachtube_stars(traces, None, 1 , 2)

    diams = time_step_diameter(traces, 300, .1)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    #plot_reachtube_stars(traces, None, 1, 2,10)

    #pre starset
    '''
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')
    '''

    #fig.show()

    #FIVE
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_05.py'
    scenario5 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car5 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    #scenario4.add_agent(car5)

    # modify mode list input

    #pre starset
    '''
    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )
    '''

    #post startset

    initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    car5.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    scenario5.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario5.add_agent(car5)

    scenario5.set_sensor(BaseStarSensor())

    start_time = time.time()

    traces = scenario5.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRA05",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })

    #plot_reachtube_stars(traces, None, 1 , 2)

    diams = time_step_diameter(traces, 300, .1)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    #plot_reachtube_stars(traces, None, 1, 2,1)
    
    #pre starset
    '''
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')
    '''

    #fig.show()


