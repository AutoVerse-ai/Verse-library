# from orbital_all_agent import OrbitalAgent
from orbital_multi_switch_agent import OrbitalAgent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
from orbital_multi_switch_sensor import OrbitalSensor

import plotly.graph_objects as go
from enum import Enum, auto
import pickle
import time
import os 

n = 0.00438138 # constant equal to (\mu/r_e^3)^{-3}
filename = "demo/aprod/refs.pkl"
filename_ahead = "demo/aprod/refs_ahead.pkl"
filename_ra = "demo/aprod/refs_ra.pkl"
filename_ahead_ra = "demo/aprod/refs_ahead_ra.pkl"
filenames = [filename, filename_ra, filename_ahead, filename_ahead_ra]

class GOMode(Enum):
    Passive = auto()
    Active = auto()

class POMode(Enum):
    Passive = auto()
    Active = auto() # in this instance, there is no obstacle so only activity should be from being close to chief

class MoveMode(Enum):
    NMT = auto()
    Inner = auto()

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

def get_trace(trace: AnalysisTree) -> np.ndarray:
    return np.array(trace.root.trace['deputy'])


if __name__ == "__main__":
    for f in filenames:
        if os.path.exists(f):
            os.remove(f)

    input_code_name = "./demo/aprod/orbital_multi_switch_controller.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    dep = OrbitalAgent("deputy", file_name=input_code_name)
    dep2 = OrbitalAgent('deputy_ahead', file_name=input_code_name)
    scenario.add_agent(dep)
    scenario.add_agent(dep2)
    orbital_sensor = OrbitalSensor()
    scenario.set_sensor(orbital_sensor)
    # modify mode list input
    # base = [10,20,0,1,2,0]
    ry = 75
    base = [0, ry+10, 0, n/2*ry*.9, 0, 0]
    x0_l = np.array(base + [base[i]-2.5 for i in range(6)] + [-2.5 for _ in range(3)] + [0 for _ in range(6)])
    x0_u = np.array(base + [base[i]+2.5 for i in range(6)] + [2.5 for _ in range(3)] + [0 for _ in range(6)])
    
    base_ahead = [4.48926-10, 74.46068+10,  0.     ,  0.04468*.9, -0.01077*1.1,  0.     ]
    x0_l_ahead = np.array(base_ahead + [base_ahead[i]-2.5 for i in range(6)] + [-2.5 for _ in range(3)] + [0 for _ in range(6)])
    x0_u_ahead = np.array(base_ahead + [base_ahead[i]+2.5 for i in range(6)] + [2.5 for _ in range(3)] + [0 for _ in range(6)])
#   ahead should start by tracking: array([ 4.48926, 74.46068,  0.     ,  0.04468, -0.01077,  0.     ])

    scenario.set_init(
        [
            [x0_l.tolist(), 
             x0_u.tolist()],
             [x0_l_ahead.tolist(), x0_u_ahead.tolist()]
        ],
        [
            # assign each agent an addition mode and state to denote whether an update occurred and priority resp.
            # actually just slightly stagger the timers 
            (GOMode.Passive, POMode.Passive, MoveMode.NMT),
            (GOMode.Passive, POMode.Passive, MoveMode.NMT),
            # (OrbitalMode.Passive,)
        ],
    )

    start = time.perf_counter()    
    trace = scenario.verify(1000, 1)
    print(f'Simulaion time: {time.perf_counter()-start:.3f}')
    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 1, 2, [1,2], plot_color=colors)
    fig.data[0].name = 'True State'
    fig.data[0].showlegend = True

    # fig = reachtube_tree(trace, None, fig, 7, 8, [7,8])
    # fig.data[-1].name = 'Est State'
    # fig.data[-1].showlegend = True

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            x_sol, u_sol = pickle.load(f)
            u_sol = np.vstack([u_sol, u_sol[-1]])
        os.remove(filename)

    if os.path.exists(filename_ra):
        with open(filename_ra, 'rb') as f:
            x_sol_ra, u_sol_ra = pickle.load(f)
            u_sol_ra = np.vstack([u_sol_ra, u_sol_ra[-1]])
        os.remove(filename_ra)

    fig.add_trace(
        go.Scatter(
            x=x_sol[:, 0],
            y=x_sol[:,1],
            mode="lines",
            line_color="#000000",
            name="Initial Reference Trajectory"
    ))

    fig.add_trace(
        go.Scatter(
            x=x_sol_ra[:, 0],
            y=x_sol_ra[:,1],
            mode="lines",
            line_color="#00AA00",
            name="Avoidance Reference Trajectory"
    ))

    fig.update_layout(
        xaxis_title='x (km)',
        yaxis_title='y (km)',
        legend_title='Trajectory Types',
    )

    fig.show()