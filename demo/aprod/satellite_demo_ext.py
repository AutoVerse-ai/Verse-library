from satellite_agent import SatelliteAgent, NoSensorSatelliteAgent, GTSatelliteAgent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
from collections import deque
from sensors import *

import plotly.graph_objects as go
from enum import Enum, auto
# import matplotlib.pyplot as plt
import time

colors = [
    # ["#CC0000", "#FF0000", "#FF3333", "#FF6666", "#FF9999", "#FFCCCC"],  # red
    ["#0000CC", "#0000FF", "#3333FF", "#6666FF", "#9999FF", "#CCCCFF"],  # blue
    ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#CCFFCC"],  # green
    ["#CCCC00", "#FFFF00", "#FFFF33", "#FFFF66", "#FFFF99", "#FFE5CC"],  # yellow
    #   ['#66CC00', '#80FF00', '#99FF33', '#B2FF66', '#CCFF99', '#FFFFCC'], # yellowgreen
    ["#CC00CC", "#FF00FF", "#FF33FF", "#FF66FF", "#FF99FF", "#FFCCFF"],  # magenta
    #   ['#00CC66', '#00FF80', '#33FF99', '#66FFB2', '#99FFCC', '#CCFFCC'], # springgreen
    ["#00CCCC", "#00FFFF", "#33FFFF", "#66FFFF", "#99FFFF", "#CCFFE5"],  # cyan
    #   ['#0066CC', '#0080FF', '#3399FF', '#66B2FF', '#99CCFF', '#CCE5FF'], # cyanblue
    ["#CC6600", "#FF8000", "#FF9933", "#FFB266", "#FFCC99", "#FFE5CC"],  # orange
      ['#6600CC', '#7F00FF', '#9933FF', '#B266FF', '#CC99FF', '#E5CCFF'], # purple
    ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#E5FFCC"],  # lime
    ["#CC0066", "#FF007F", "#FF3399", "#FF66B2", "#FF99CC", "#FFCCE5"],  # pink
]

n = 0.00438138 # constant equal to (\mu/r_e^3)^{-3}

class SatelliteMode(Enum):
    Passive = auto()

#assuming scalar first as in trace, n x 4 array
def quat_to_rot_vect(q: np.ndarray) -> np.ndarray:
    q0 = q[:,0]
    rho = q[:,1:]

    theta = 2*np.arccos(np.clip(q0, -1, 1))
    sin_half_theta = np.sqrt(np.clip(1-q0**2, 0, 1))

    sin_half_theta[sin_half_theta<1e-8] = 1 # filler value, don't care about anything with sin values too low
    axis = rho / sin_half_theta.reshape(-1, 1)
    axis[sin_half_theta<1e-8] = 0
    return theta.reshape(-1, 1) * axis

def get_trace(tree: Union[AnalysisTree, AnalysisTreeNode]) -> np.ndarray:
    return np.array(tree.root.trace['deputy'])

def get_final_states_sim(n: AnalysisTreeNode) -> Tuple[List]: 
    return n.trace['deputy'][-1] # in general return a dict of the traces, interating over/key'd by agent_id

if __name__ == "__main__":
    input_code_name = "./demo/aprod/satellite_controller.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    dep = NoSensorSatelliteAgent("deputy", file_name=input_code_name)
    scenario.add_agent(dep)
    T = 2000
    dt = 0.1 
    dT = 60 # if less than dt, then update sensor, call controller every dt time
    dT = dt if dT < dt else dT
    N = 1

    scenario.set_init(
        [
            [[0, -400, 0, 400*n, 0, 0, 0, -400, 0, 400*n, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0.05], 
             [0, -400, 0, 400*n, 0, 0, 0, -400, 0, 400*n, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0.05]],
        ],
        [(SatelliteMode.Passive,)],
    )
    start = time.perf_counter()
    trace = scenario.simulate(dT, dt)
    id = 1+trace.root.id
    queue = deque()
    queue.append(trace.root) 
    while len(queue):
        cur_node = queue.popleft()
        own_state = get_final_states_sim(cur_node)
        sensed_state = combine_sensors(own_state[1:]) # TODO: make own sensor class at some point that's separate from existing sensor class
        scenario.set_init(
            [[sensed_state, sensed_state]], # this should eventually be a range 
            [(SatelliteMode.Passive,)]
        )
        id += 1
        new_trace = scenario.simulate(dT, dt)
        temp_root = new_trace.root
        new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + dT, id)
        cur_node.child.append(new_node)
        if new_node.start_time + dT>=T: # if the time of the current simulation + start_time is at or above total time, don't add
            continue
        queue.append(new_node)
    trace.nodes = trace._get_all_nodes(trace.root)

    fig = go.Figure()
    fig: go.Figure = simulation_tree(trace, None, fig, 1, 2, [1, 2], "lines", "trace")
    fig: go.Figure = simulation_tree(trace, None, fig, 7, 8, [7, 8], "lines", "trace", plot_color=colors)

    fig.data[-2].name = 'Ground truth' 
    fig.data[-1].name = 'Sampled traces'
    fig.data[-2].showlegend = True
    fig.data[-1].showlegend = True

    fig.update_layout(
        legend_title_text="Trace types",
        xaxis_title="x",
        yaxis_title="y"
    )
    fig.show()
    print(f'Total time for {N} simulations: {int((time.perf_counter()-start)//60)}\'{(time.perf_counter()-start)%60:.2f}\"')