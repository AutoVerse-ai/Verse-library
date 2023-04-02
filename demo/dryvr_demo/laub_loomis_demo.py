from origin_agent import laub_loomis_agent
from verse import Scenario
from verse.plotter.plotter2D import *
import time
import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()
#traces.nodes[0].trace['car1'][-2:]
if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/laub_loomis_controller.py'
    scenario = Scenario()

    car = laub_loomis_agent('car1', file_name=input_code_name)
    scenario.add_agent(car)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    W  = .1
    scenario.set_init(
        [
            [[1.2-W, 1.05-W, 1.5-W, 2.4-W, 1-W,.1 -W,.45-W, W], [1.2+W, 1.05+W, 1.5+W, 2.4+W, 1+W,.1 +W,.45+W,W]],
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ]
    )
    start_time = time.time()

    traces = scenario.verify(20, 0.02,params={"bloating_method":"GLOBAL"})
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 4, [1, 2], 'lines', 'trace', combine_rect=3)
    run_time = time.time() - start_time
    print({
        "tool": "verse",
        "benchmark": "LALO20",
        "setup": "W01",
        "result": "1",
        "time": run_time,
        "metric2": str(traces.nodes[0].trace['car1'][-1][4] - traces.nodes[0].trace['car1'][-2][4]  ),
        "metric3": "n/a",
    })


    scenario1 = Scenario()

    car1 = laub_loomis_agent('car1', file_name=input_code_name)
    scenario1.add_agent(car1)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    W = .05
    scenario1.set_init(
        [
            [[1.2 - W, 1.05 - W, 1.5 - W, 2.4 - W, 1 - W, .1 - W, .45 - W,W],
             [1.2 + W, 1.05 + W, 1.5 + W, 2.4 + W, 1 + W, .1 + W, .45 + W,W]],
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ]
    )
    start_time = time.time()

    traces = scenario1.verify(20, 0.02, params={"bloating_method": "GLOBAL"})
    run_time = time.time() - start_time
    print({
        "tool": "verse",
        "benchmark": "LALO20",
        "setup": "W005",
        "result": "1",
        "time": run_time,
        "metric2": str(traces.nodes[0].trace['car1'][-1][4] - traces.nodes[0].trace['car1'][-2][4]),
        "metric3": "n/a",
    })


    # fig = reachtube_tree(traces, None, fig, 0, 4, [1, 2], 'lines', 'trace', combine_rect=3)
    #
    # scenario2 = Scenario()
    #
    # car2 = laub_loomis_agent('car1', file_name=input_code_name)
    # scenario2.add_agent(car2)
    # # car = vanderpol_agent('car2', file_name=input_code_name)
    # # scenario.add_agent(car)
    # # scenario.set_sensor(FakeSensor2())
    # # modify mode list input
    # W = .01
    # scenario2.set_init(
    #     [
    #         [[1.2 - W, 1.05 - W, 1.5 - W, 2.4 - W, 1 - W, .1 - W, .45 - W,W],
    #          [1.2 + W, 1.05 + W, 1.5 + W, 2.4 + W, 1 + W, .1 + W, .45 + W,W]],
    #     ],
    #     [
    #         tuple([AgentMode.Default]),
    #         # tuple([AgentMode.Default]),
    #     ]
    # )
    # traces = scenario2.verify(20, 0.02, params={"bloating_method": "GLOBAL"})
    # fig = reachtube_tree(traces, None, fig, 0, 4, [1, 2], 'lines', 'trace', combine_rect=3)
    # fig.update_layout(
    #     xaxis_title="t", yaxis_title="x4"
    # )
    # fig.show()

    # scenario3 = Scenario()
    #
    # car3 = laub_loomis_agent('car1', file_name=input_code_name)
    # scenario3.add_agent(car3)
    # # car = vanderpol_agent('car2', file_name=input_code_name)
    # # scenario.add_agent(car)
    # # scenario.set_sensor(FakeSensor2())
    # # modify mode list input
    # W = .1
    # scenario3.set_init(
    #     [
    #         [[1.2 - W/4.5, 1.05 - W/2, 1.5 - W/16, 2.4 + W/3, 1 - W/7, .1 + W/1, .45 - W/2.5, W]],
    #     ],
    #     [
    #         tuple([AgentMode.Default]),
    #         # tuple([AgentMode.Default]),
    #     ]
    # )
    #
    # traces = scenario3.simulate(20, 0.02)
    #
    # fig = simulation_tree(traces, None, fig, 0, 4, [0, 4],
    #                       'lines', 'trace')
    # fig.show()
