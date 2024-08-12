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


class CraftMode(Enum):
    Approaching = auto()
    Rendezvous = auto()
    Aborting = auto()


if __name__ == "__main__":

    

    # #SIX
    # input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_06.py'
    # scenario6 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    # car6 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    # #scenario6.add_agent(car6)

    # #pre starset
    # '''
    # # modify mode list input

    # scenario6.set_init(
    #     [
    #         [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
    #     ],
    #     [
    #         tuple([CraftMode.Approaching]),
    #     ]
    # )
    # '''

    # #post startset

    # initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    # car6.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    # scenario6.config.reachability_method = ReachabilityMethod.STAR_SETS
    # scenario6.add_agent(car6)

    # scenario6.set_sensor(BaseStarSensor())


    # start_time = time.time()

    # traces = scenario6.verify(300, .1)
    # run_time = time.time() - start_time

    # print({
    #     "tool": "verse",
    #     "benchmark": "Rendezvous",
    #     "setup": "SRA06",
    #     "result": "1",
    #     "time": run_time,
    #     "metric2": "n/a",
    #     "metric3": "n/a",
    # })

    # import plotly.graph_objects as go
    # from verse.plotter.plotterStar import *

    # #plot_reachtube_stars(traces, None, 1, 2,1)

    # diams = time_step_diameter(traces, 300, .10)
    # print(len(diams))
    # print(sum(diams))
    # print(diams[0])
    # print(diams[-1])


    # #pre startset
    # '''
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
    #                      'lines', 'trace')
    # '''

    # #fig.show()

    # #SEVEN
    # input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_07.py'
    # scenario7 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    # car7 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    # #scenario4.add_agent(car7)

    # #pre starset
    # '''
    # # modify mode list input

    # scenario7.set_init(
    #     [
    #         [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
    #     ],
    #     [
    #         tuple([CraftMode.Approaching]),
    #     ]
    # )
    # '''

    # #post startset

    # initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    # car7.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    # scenario7.config.reachability_method = ReachabilityMethod.STAR_SETS
    # scenario7.add_agent(car7)

    # scenario7.set_sensor(BaseStarSensor())

    # start_time = time.time()

    # traces = scenario7.verify(300, .1)
    # run_time = time.time() - start_time

    # print({
    #     "tool": "verse",
    #     "benchmark": "Rendezvous",
    #     "setup": "SRA07",
    #     "result": "1",
    #     "time": run_time,
    #     "metric2": "n/a",
    #     "metric3": "n/a",
    # })

    # import plotly.graph_objects as go
    # from verse.plotter.plotterStar import *

    # #plot_reachtube_stars(traces, None, 1, 2,10)

    # diams = time_step_diameter(traces, 300, .1)
    # print(len(diams))
    # print(sum(diams))
    # print(diams[0])
    # print(diams[-1])

    # #pre starset
    # '''
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
    #                      'lines', 'trace')
    # '''

    # #fig.show()

    # #EIGHT
    # input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_08.py'
    # scenario8 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    # car8 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    # #scenario8.add_agent(car8)


    # #pre starset
    # '''
    # # modify mode list input

    # scenario8.set_init(
    #     [
    #         [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
    #     ],
    #     [
    #         tuple([CraftMode.Approaching]),
    #     ]
    # )
    # '''

    # #post startset

    # initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    # car8.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    # scenario8.config.reachability_method = ReachabilityMethod.STAR_SETS
    # scenario8.add_agent(car8)

    # scenario8.set_sensor(BaseStarSensor())

    # start_time = time.time()

    # traces = scenario8.verify(300, .1)
    # run_time = time.time() - start_time

    # print({
    #     "tool": "verse",
    #     "benchmark": "Rendezvous",
    #     "setup": "SRA08",
    #     "result": "1",
    #     "time": run_time,
    #     "metric2": "n/a",
    #     "metric3": "n/a",
    # })

    # diams = time_step_diameter(traces, 300, .1)
    # print(len(diams))
    # print(sum(diams))
    # print(diams[0])
    # print(diams[-1])


    # import plotly.graph_objects as go
    # from verse.plotter.plotterStar import *

    # #plot_reachtube_stars(traces, None, 1, 2,10)

    # #pre starset
    # '''
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
    #                      'lines', 'trace')
    # '''

    # #fig.show()

    # #NINE
    # input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_U01.py'
    # scenario9 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    # car9 = spacecraft_linear_agent('test', file_name=input_code_name)
    # #scenario9.add_agent(car9)

    # #pre starset
    # '''
    # # modify mode list input

    # scenario9.set_init(
    #     [
    #         [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
    #     ],
    #     [
    #         tuple([CraftMode.Approaching]),
    #     ]
    # )
    # '''

    # #post startset

    # initial_set_polytope = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])
    # car9.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    # scenario9.config.reachability_method = ReachabilityMethod.STAR_SETS
    # scenario9.add_agent(car9)

    # scenario9.set_sensor(BaseStarSensor())

    # start_time = time.time()

    # traces = scenario9.verify(300, .1)
    # run_time = time.time() - start_time

    # print({
    #     "tool": "verse",
    #     "benchmark": "Rendezvous",
    #     "setup": "SRU01",
    #     "result": "1",
    #     "time": run_time,
    #     "metric2": "n/a",
    #     "metric3": "n/a",
    # })

    # import plotly.graph_objects as go
    # from verse.plotter.plotterStar import *

    # #plot_reachtube_stars(traces, None, 1, 2,10)

    # diams = time_step_diameter(traces, 300, .1)
    # print(len(diams))
    # print(sum(diams))
    # print(diams[0])
    # print(diams[-1])


    # #pre starset
    # '''
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
    #                      'lines', 'trace')
    # '''

    # #fig.show()

    #TEN
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_U02.py'
    scenario10 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car10 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    scenario10.add_agent(car10)

    

    #post starset
    '''
    # modify mode list input

    scenario10.set_init(
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
    car10.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([CraftMode.Approaching]))

    scenario10.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario10.add_agent(car10)

    scenario10.set_sensor(BaseStarSensor())

    start_time = time.time()

    traces = scenario10.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRU02",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    #plot_reachtube_stars(traces, None, 1, 2,10)

    # diams = time_step_diameter(traces, 300, .1)
    # print(len(diams))
    # print(sum(diams))
    # print(diams[0])
    # print(diams[-1])

    
    #pre starset
    '''
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    #fig.show()
    '''
