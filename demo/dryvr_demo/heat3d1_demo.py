from origin_agent_heat import heat3d1_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto
import time

class CraftMode(Enum):
    Default = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/heat3d1_controller.py'
    #scenario = Scenario()
    scenario = Scenario(ScenarioConfig(parallel=False))


    car = heat3d1_agent('test', file_name=input_code_name)
    scenario.add_agent(car)

    # modify mode list input
    scenario.set_init(
        [
            [[.9,.9,.9,0,0,.9,.9,.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.9,.9,.9,0,0,.9,.9,.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1.1,1.1,1.1,0,0,1.1,1.1,1.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.1,1.1,1.1,0,0,1.1,1.1,1.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        ],
        [
            tuple([CraftMode.Default]),
        ]
    )

    init = np.array([[.9,.9,.9,0,0,.9,.9,.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.9,.9,.9,0,0,.9,.9,.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1.1,1.1,1.1,0,0,1.1,1.1,1.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.1,1.1,1.1,0,0,1.1,1.1,1.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    ID = np.linalg.norm(init[1,:]-init[0,:], ord=1)
    # scenario.config.parallel = False


    start_time = time.time()

    traces = scenario.verify(40, 0.02, params = {"sim_trace_num":35})
    run_time = time.time() - start_time

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
        "benchmark": "Heat3D",
        "setup": "HEAT01",
        "result": "1",
        "time": run_time,
        "metric2" : str(FD/ID),
        "metric3": str(AD/ID),
    })
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 0, 63, [0, 63],
                         'lines', 'trace')
    # fig.add_trace(go.Scatter(x=[
    #         1+0.157,1+0.033,1+-0.033,1+-0.157,1+-0.157,1+-0.033,1+0.033,1+0.157,1+0.157
    #     ],
    #     y=[
    #         1+0.033,1+0.157,1+0.157,1+0.033,1+-0.033,1+-0.157,1+-0.157,1+-0.033,1+0.033
    #     ]
    # ))
    fig.update_layout(
        xaxis_title="t", yaxis_title="x62"
    )
    fig.update_xaxes(
        range=[0,40])
    #fig.show()
