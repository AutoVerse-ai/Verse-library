from origin_agent import gearbox_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.scenario import ScenarioConfig
import plotly.graph_objects as go
from enum import Enum, auto
import time

class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/sleeve_controller.py'
    config=ScenarioConfig(init_seg_length=1, parallel=False)
    scenario = Scenario(config=config)

    car = gearbox_agent('sleeve', file_name=input_code_name)
    scenario.add_agent(car)

    scenario.set_init(
        [
            [[-0.0168, 0.0029, 0, 0, 0,0,1], [-0.0166, 0.0031, 0, 0, 0,0,1]],
        ],
        [
            tuple([AgentMode.Free]),
        ]
    )
    start_time = time.time()
    traces = scenario.verify(.1, 1e-4)
    run_time = time.time() - start_time

    init = np.array([[-0.0168, 0.0029, 0, 0, 0,0,1], [-0.0166, 0.0031, 0, 0, 0,0,1]])
    ID = np.linalg.norm(init[1,:]-init[0,:], ord=1)

    Frect = np.array(traces.nodes[-1].trace['sleeve'])[-2:,1:]
    FD = np.linalg.norm(Frect[1,:]-Frect[0,:], ord=1)

    num = 0
    total_sum = 0
    for node in traces.nodes:
        trace = np.array(traces.nodes[-1].trace['sleeve'])[:,1:]
        lb = trace[::2,:]
        ub = trace[1::2,:]
        trace_diag = np.linalg.norm(lb-ub, axis=1, ord=1)
        num = num + len(trace_diag)
        total_sum = total_sum + np.sum(trace_diag)

    AD = total_sum/num

    print({
        "tool": "verse",
        "benchmark": "Gear",
        "setup": "GRBX01",
        "result": "0",
        "time": run_time,
        "metric2": str(FD/ID),
        "metric3": str(AD/ID),
    })
    traces.dump('./demo/gearbox/output.json')
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], 'lines', 'trace', sample_rate=1)
    fig.show()



    # input_code_name = './demo/dryvr_demo/sleeve_controller.py'
    # config = ScenarioConfig(init_seg_length=1, parallel=False)
    # scenario = Scenario(config=config)

    # car = gearbox_agent('sleeve', file_name=input_code_name)
    # scenario.add_agent(car)

    # scenario.set_init(
    #     [
    #         [[-0.01675,0.00285 , 0, 0, 0, 0, 1], [-0.01665,0.00315, 0, 0, 0, 0, 1]],
    #     ],
    #     [
    #         tuple([AgentMode.Free]),
    #     ]
    # )
    # start_time = time.time()
    # traces = scenario.verify(.11, 1e-4)
    # run_time = time.time() - start_time

    # print({
    #     "tool": "verse",
    #     "benchmark": "Gear",
    #     "setup": "GRBX02",
    #     "result": "0",
    #     "time": run_time,
    #     "metric2": 'n/a',
    #     "metric3": "n/a",
    # })
    # traces.dump('./demo/gearbox/output.json')
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], 'lines', 'trace', sample_rate=1)
    # fig.show()
    # # fig.show()
