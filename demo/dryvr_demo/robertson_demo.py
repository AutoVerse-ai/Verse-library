from origin_agent import robertson_agent
from verse import Scenario
from verse.plotter.plotter2D import *

import plotly.graph_objects as go
from enum import Enum, auto
import time

class AgentMode(Enum):
    Default = auto()
import os

def compute_xyz(tree):
    # return tree
    trace = np.array(list(tree.nodes[0].trace.values())[0])
    agent_id = list(tree.nodes[0].trace.keys())[0]
    x_low = trace[0::2,1]
    x_high = trace[1::2,1]
    y_low = trace[0::2,2]
    y_high = trace[1::2,2]
    z_low = trace[0::2,3]
    z_high = trace[1::2,3]
    xyz_low = x_low+y_low+z_low
    xyz_high = x_high+y_high+z_high 
    trace_new = np.zeros((trace.shape[0],5))
    trace_new[:,0:4] = trace 
    trace_new[0::2,4] = xyz_low 
    trace_new[1::2,4] = xyz_high
    tree.nodes[0].trace[agent_id] = trace_new.tolist()
    return tree

if __name__ == "__main__":
    print(os.getcwd())
    input_code_name = './demo/dryvr_demo/robertson_controller.py'
    fig = go.Figure()

    scenario1 = Scenario()
    car1 = robertson_agent('car1', file_name=input_code_name, beta = 1e3, gamma=1e7)
    scenario1.add_agent(car1)
    scenario1.set_init(
        [
            [[1, 0, 0], [1, 0, 0]],
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ]
    )
    start_time = time.time()

    traces1 = scenario1.verify(40, .1)
    traces1 = compute_xyz(traces1)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "ROBE21",
        "setup": "B3G7",
        "result": "1",
        "time": run_time,
        "metric2": traces1.nodes[0].trace['car1'][-1][1] + traces1.nodes[0].trace['car1'][-1][2] +
                   traces1.nodes[0].trace['car1'][-1][3] - (
                           traces1.nodes[0].trace['car1'][-2][1] + traces1.nodes[0].trace['car1'][-2][2] +
                           traces1.nodes[0].trace['car1'][-2][3]),
        "metric3": "n/a",
    })
    fig = reachtube_tree(traces1, None, fig, 0, 4, [0, 1], 'lines', 'trace', combine_rect=3)
    scenario = Scenario()
    car = robertson_agent('car1', file_name=input_code_name, beta = 1e2, gamma=1e3)
    scenario.add_agent(car)
    scenario.set_init(
        [
            [[1, 0, 0], [1, 0, 0]],
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ]
    )
    start_time = time.time()

    traces = scenario.verify(40, .1)
    traces = compute_xyz(traces)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "ROBE21",
        "setup": "B2G3",
        "result": "1",
        "time": run_time,
        "metric2": traces.nodes[0].trace['car1'][-1][1] + traces.nodes[0].trace['car1'][-1][2] +
                   traces.nodes[0].trace['car1'][-1][3] - (
                               traces.nodes[0].trace['car1'][-2][1] + traces.nodes[0].trace['car1'][-2][2] +
                               traces.nodes[0].trace['car1'][-2][3]),
        "metric3": "n/a",
    })
    fig = reachtube_tree(traces, None, fig, 0, 4, [0, 1], 'lines', 'trace', combine_rect=3)

    scenario2 = Scenario()
    car2 = robertson_agent('car1', file_name=input_code_name, beta = 1e3, gamma=1e5)
    scenario2.add_agent(car2)
    scenario2.set_init(
        [
            [[1, 0, 0], [1, 0, 0]],
        ],
        [
            tuple([AgentMode.Default]),
            # tuple([AgentMode.Default]),
        ]
    )
    start_time = time.time()

    traces2 = scenario2.verify(40, .1)

    traces2 = compute_xyz(traces2)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "ROBE21",
        "setup": "B3G5",
        "result": "1",
        "time": run_time,
        "metric2": traces2.nodes[0].trace['car1'][-1][1] + traces2.nodes[0].trace['car1'][-1][2] + traces2.nodes[0].trace['car1'][-1][3]  - (traces2.nodes[0].trace['car1'][-2][1] + traces2.nodes[0].trace['car1'][-2][2] + traces2.nodes[0].trace['car1'][-2][3]) ,
        "metric3": "n/a",
    })
    fig = reachtube_tree(traces2, None, fig, 0, 4, [0, 1], 'lines', 'trace', combine_rect=3)
    fig.update_layout(
        xaxis_title="t", yaxis_title="s"
    )



    fig.show()
