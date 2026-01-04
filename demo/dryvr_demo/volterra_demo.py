from origin_agent import volterra_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto
import time

class CraftMode(Enum):
    inside = auto()
    outside = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/volterra_controller.py'
    #scenario = Scenario()
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))


    car = volterra_agent('test', file_name=input_code_name)
    scenario.add_agent(car)

    # modify mode list input
    e = .012
    scenario.set_init(
        [
            [[1.3 - e, 1, 0], [1.3 + e, 1, 0]],
        ],
        [
            tuple([CraftMode.outside]),
        ]
    )

    # traces = scenario.simulate(3.64, .01)
    # fig = go.Figure()
    # fig = simulation_anime(traces, None, fig, 1, 2, [
    #                        1, 2], 'lines', 'trace', sample_rate=1)
    # fig.show()


    start_time = time.time()

    traces = scenario.verify(3.64, .01)
    run_time = time.time() - start_time


    init = np.array([[1.3 - e, 1, 0], [1.3 + e, 1, 0]])
    ID = np.linalg.norm(init[1,:]-init[0,:], ord=1)

    Frect = np.array(traces.nodes[-1].trace['test'])[-2:,1:]
    FD = np.linalg.norm(Frect[1,:]-Frect[0,:], ord=1)

    num = 0
    total_sum = 0
    for node in traces.nodes:
        trace = np.array(traces.nodes[-1].trace['test'])[:,1:]
        lb = trace[::2,:]
        ub = trace[1::2,:]
        trace_diag = np.linalg.norm(lb-ub, axis=1, ord=1)
        num = num + len(trace_diag)
        total_sum = total_sum + np.sum(trace_diag)

    AD = total_sum/num

    print({
        "tool": "verse",
        "benchmark": "LOVO21",
        "setup": "",
        "result": "1",
        "time": run_time,
        "metric2": str(FD/ID),
        "metric3": str(AD/ID),
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')
    fig.add_trace(go.Scatter(x=[
        1 + 0.161, 1 + 0.06669, 1 + -0.06669, 1 + -0.161, 1 + -0.161, 1 + -0.06669, 1 + 0.06669, 1 + 0.161, 1 + 0.161
    ],
        y=[
            1 + 0.06669, 1 + 0.161, 1 + 0.161, 1 + 0.06669, 1 + -0.06669, 1 + -0.161, 1 + -0.161, 1 + -0.06669,
            1 + 0.06669
        ]
    ))
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=1 - .161, y0=1 - .161, x1=1 + .161, y1=1 + .161,
                  line_color="LightSeaGreen",
                  )

    fig.show()

