from origin_agent_spacecraft import spacecraft_linear_agent, spacecraft_linear_agent_nd
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor
import time
import plotly.graph_objects as go
from enum import Enum, auto


class CraftMode(Enum):
    Approaching = auto()
    Rendezvous = auto()
    Aborting = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller.py'
#     scenario = Scenario()
#
#     car = spacecraft_linear_agent('test', file_name=input_code_name)
#     scenario.add_agent(car)
#
#     # modify mode list input
#
#     scenario.set_init(
#         [
#             [[-925, -425, 0, 0,0,0], [-875, -375, 0, 0, 0,0]],
#         ],
#         [
#             tuple([CraftMode.Approaching]),
#         ]
#     )
#
#     # traces = scenario.simulate(200, 1)
#     # fig = go.Figure()
#     # fig = simulation_anime(traces, None, fig, 1, 2, [
#     #                        1, 2], 'lines', 'trace', sample_rate=1)
#     # fig.show()
#
#
#     start_time = time.time()
#
#     traces = scenario.verify(300, .1)
#     run_time = time.time() - start_time
#
#     print({
#         "tool": "verse",
#         "benchmark": "Rendezvous",
#         "setup": "SRA01",
#         "result": "1",
#         "time": run_time,
#         "metric2": "n/a",
#         "metric3": "n/a",
#     })
#     fig = go.Figure()
#     fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
#                          'lines', 'trace')
#     fig.update_layout(
#         xaxis_title="x", yaxis_title="y"
#     )
#
#     fig.show()
#
#     input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_na.py'
#     scenario1 = Scenario()
#     car1 = spacecraft_linear_agent('test', file_name=input_code_name)
#     scenario1.add_agent(car1)
#
#     # modify mode list input
#
#     scenario1.set_init(
#         [
#             [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
#         ],
#         [
#             tuple([CraftMode.Approaching]),
#         ]
#     )
#
#
#
#     start_time = time.time()
#
#     traces = scenario1.verify(300, .1)
#     run_time = time.time() - start_time
#
#     print({
#         "tool": "verse",
#         "benchmark": "Rendezvous",
#         "setup": "SRNA01",
#         "result": "1",
#         "time": run_time,
#         "metric2": "n/a",
#         "metric3": "n/a",
#     })
#     fig = go.Figure()
#     fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
#                          'lines', 'trace')
#     fig.update_layout(
#         xaxis_title="x", yaxis_title="y"
#     )
#     fig.show()
#
#
#     input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_02.py'
#     scenario2 = Scenario()
#     car2 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
#     scenario2.add_agent(car2)
#
#     # modify mode list input
#
#     scenario2.set_init(
#         [
#             [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
#         ],
#         [
#             tuple([CraftMode.Approaching]),
#         ]
#     )
#
#
#
#     start_time = time.time()
#
#     traces = scenario2.verify(300, .1)
#     run_time = time.time() - start_time
#
#     print({
#         "tool": "verse",
#         "benchmark": "Rendezvous",
#         "setup": "SRA02",
#         "result": "1",
#         "time": run_time,
#         "metric2": "n/a",
#         "metric3": "n/a",
#     })
#     fig = go.Figure()
#     fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
#                          'lines', 'trace')
#
#     fig.show()
#
#
#     input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_03.py'
#     scenario3 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
#     car3 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
#     scenario3.add_agent(car3)
#
#     # modify mode list input
#
#     scenario3.set_init(
#         [
#             [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
#         ],
#         [
#             tuple([CraftMode.Approaching]),
#         ]
#     )
#
#
#
#     start_time = time.time()
#
#     traces = scenario3.verify(300, .1)
#     run_time = time.time() - start_time
#
#     print({
#         "tool": "verse",
#         "benchmark": "Rendezvous",
#         "setup": "SRA03",
#         "result": "1",
#         "time": run_time,
#         "metric2": "n/a",
#         "metric3": "n/a",
#     })
#     fig = go.Figure()
#     fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
#                          'lines', 'trace')
#
#     fig.show()
#
#     input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_04.py'
#     scenario4 = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
#     car4 = spacecraft_linear_agent('test', file_name=input_code_name)
#     scenario4.add_agent(car4)
#
#     # modify mode list input
#
#     scenario4.set_init(
#         [
#             [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
#         ],
#         [
#             tuple([CraftMode.Approaching]),
#         ]
#     )
#
#     start_time = time.time()
#
#     traces = scenario4.verify(300, .1)
#     run_time = time.time() - start_time
#
#     print({
#         "tool": "verse",
#         "benchmark": "Rendezvous",
#         "setup": "SRA04",
#         "result": "1",
#         "time": run_time,
#         "metric2": "n/a",
#         "metric3": "n/a",
#     })
#     fig = go.Figure()
#     fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
#                          'lines', 'trace')
#
#     fig.show()
#
#     input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_05.py'
#     scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
#     car4 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
#     scenario4.add_agent(car4)
#
#     # modify mode list input
#
#     scenario4.set_init(
#         [
#             [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
#         ],
#         [
#             tuple([CraftMode.Approaching]),
#         ]
#     )
#
#     start_time = time.time()
#
#     traces = scenario4.verify(300, .1)
#     run_time = time.time() - start_time
#
#     print({
#         "tool": "verse",
#         "benchmark": "Rendezvous",
#         "setup": "SRA05",
#         "result": "1",
#         "time": run_time,
#         "metric2": "n/a",
#         "metric3": "n/a",
#     })
#     fig = go.Figure()
#     fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
#                          'lines', 'trace')
#
#     fig.show()
#
#     input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_06.py'
#     scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
#     car4 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
#     scenario4.add_agent(car4)
#
#     # modify mode list input
#
#     scenario4.set_init(
#         [
#             [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
#         ],
#         [
#             tuple([CraftMode.Approaching]),
#         ]
#     )
#
#     start_time = time.time()
#
#     traces = scenario4.verify(300, .1)
#     run_time = time.time() - start_time
#
#     print({
#         "tool": "verse",
#         "benchmark": "Rendezvous",
#         "setup": "SRA06",
#         "result": "1",
#         "time": run_time,
#         "metric2": "n/a",
#         "metric3": "n/a",
#     })
#     fig = go.Figure()
#     fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
#                          'lines', 'trace')
#
#     fig.show()
#
#     input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_07.py'
#     scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
#     car4 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
#     scenario4.add_agent(car4)
#
#     # modify mode list input
#
#     scenario4.set_init(
#         [
#             [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
#         ],
#         [
#             tuple([CraftMode.Approaching]),
#         ]
#     )
#
#     start_time = time.time()
#
#     traces = scenario4.verify(300, .1)
#     run_time = time.time() - start_time
#
#     print({
#         "tool": "verse",
#         "benchmark": "Rendezvous",
#         "setup": "SRA07",
#         "result": "1",
#         "time": run_time,
#         "metric2": "n/a",
#         "metric3": "n/a",
#     })
#     fig = go.Figure()
#     fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
#                          'lines', 'trace')
#
#     fig.show()

    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_08.py'
    scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car4 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    scenario4.add_agent(car4)

    # modify mode list input

    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    start_time = time.time()

    traces = scenario4.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRA08",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    fig.show()

    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_U01.py'
    scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car4 = spacecraft_linear_agent('test', file_name=input_code_name)
    scenario4.add_agent(car4)

    # modify mode list input

    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    start_time = time.time()

    traces = scenario4.verify(300, .1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "Rendezvous",
        "setup": "SRU01",
        "result": "1",
        "time": run_time,
        "metric2": "n/a",
        "metric3": "n/a",
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    fig.show()

    input_code_name = './demo/dryvr_demo/spacecraft_linear_controllers/spacecraft_linear_controller_U02.py'
    scenario4 = Scenario(ScenarioConfig(init_seg_length=10, parallel=False))
    car4 = spacecraft_linear_agent_nd('test', file_name=input_code_name)
    scenario4.add_agent(car4)

    # modify mode list input

    scenario4.set_init(
        [
            [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
        ],
        [
            tuple([CraftMode.Approaching]),
        ]
    )

    start_time = time.time()

    traces = scenario4.verify(300, .1)
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
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')

    fig.show()
