from orbital_agent import OrbitalAgent, OpenOrbitalAgent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *

import plotly.graph_objects as go
from enum import Enum, auto
import pickle
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
    #   ['#6600CC', '#7F00FF', '#9933FF', '#B266FF', '#CC99FF', '#E5CCFF'], # purple
    ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#E5FFCC"],  # lime
    # ["#CC0066", "#FF007F", "#FF3399", "#FF66B2", "#FF99CC", "#FFCCE5"],  # pink
]

n = 0.00438138 # constant equal to (\mu/r_e^3)^{-3}

class OrbitalMode(Enum):
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

def get_trace(trace: AnalysisTree) -> np.ndarray:
    return np.array(trace.root.trace['deputy'])

if __name__ == "__main__":
    input_code_name = "./demo/aprod/orbital_controller.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    dep = OrbitalAgent("deputy", file_name=input_code_name)
    scenario.add_agent(dep)
    # scenario.set_sensor(CraftSensor())
    # modify mode list input
    base = [10,20,0,1,2,0]
    x0 = np.array(base + [np.random.rand()*5-2.5+base[i] for i in range(3)] + base[3:])
    # x0 = np.array(base + [np.random.rand()*0.5-0.25+base[i] for i in range(6)])
    scenario.set_init(
        [
            [x0, 
             x0],
        ],
        [
            (OrbitalMode.Passive,)
        ],
    )

    start = time.perf_counter()    
    trace = scenario.simulate(2000, 1)
    print(f'Simulaion time: {time.perf_counter()-start:.3f}')
    fig = go.Figure()
    fig = simulation_tree(trace, None, fig, 1, 2, [1,2], plot_color=[['#0000CC']])
    fig = simulation_tree(trace, None, fig, 7, 8, [7,8])
    x_sol = None
    with open('demo/aprod/ref_traj.pkl', 'rb') as f:
        x_sol = pickle.load(f)
    fig.add_trace(
        go.Scatter(
            x=x_sol[:, 0],
            y=x_sol[:,1],
            mode="lines",
            line_color="#000000",
            name="Reference Trajectory"
    ))
    fig.update_layout(
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        legend_title='Trajectory Types',
    )
    fig.data[0].name = 'True State'
    fig.data[1].name = 'Est State'

    final_state = get_trace(trace)[-1][1:]
    open_scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    open_dep = OpenOrbitalAgent("deputy")
    open_scenario.add_agent(open_dep)
    open_scenario.set_init(
        [[
            final_state, final_state
        ]],
        [
            (OrbitalMode.Passive,)
        ]
    )
    open_trace = open_scenario.simulate(6000, 1)
    fig = simulation_tree(open_trace, None, fig, 1, 2, [1,2], plot_color=[['#0000CC']])
    fig = simulation_tree(open_trace, None, fig, 7, 8, [7,8])
    fig.data[-2].name = 'True State (u=0)'
    fig.data[-1].name = 'Est State (u=0)'
    
    for trace in fig.data:
        trace.showlegend = True 
    fig.show()