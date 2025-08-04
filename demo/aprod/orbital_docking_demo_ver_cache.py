# from orbital_all_agent import OrbitalAgent
from orbital_docking_agent_cache import OrbitalAgent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
# from orbital_docking_sensor_v2 import OrbitalSensor
from orbital_docking_sensor_proxdyn import OrbitalSensor

import plotly.graph_objects as go
from enum import Enum, auto
import pickle
import time
import os, shutil

n = 0.00438138 # constant equal to (\mu/r_e^3)^{-3}
filename = "demo/aprod/refs.pkl"
t_global = "demo/aprod/time.pkl" # write in the final time of current traj, check with initial set time
sim_count = "demo/aprod/sim_count.pkl" # use in conjunction with base_final to read and write in final states of nominal trajs, reset once time_global is updated
cache_base = "demo/aprod/cached_traj"
last_mode = "demo/aprod/last_mode.pkl" # keep track of the last prox mode so strategy only used when going prox passive -> active and vice versa (keep ground sensor as is for time being)
num_trajs = "demo/aprod/num_trajs.pkl"

files = [filename, t_global, sim_count, last_mode, num_trajs]

class GOMode(Enum):
    Passive = auto()
    Active = auto()

class POMode(Enum):
    Passive = auto()
    Active = auto()

class MoveMode(Enum):
    NMT = auto()
    Docking = auto()

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
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(cache_base):
        shutil.rmtree(cache_base)

    # input_code_name = "./demo/aprod/orbital_docking_controller.py"
    # input_code_name = "./demo/aprod/orbital_docking_controller_v2.py"
    input_code_name = "./demo/aprod/orbital_docking_controller_proxdyn.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    dep = OrbitalAgent("deputy", file_name=input_code_name)
    scenario.add_agent(dep)
    orbital_sensor = OrbitalSensor()
    scenario.set_sensor(orbital_sensor)
    # modify mode list input
    # base = [10,20,0,1,2,0]
    T = 5500
    ry = 75
    base = [0, ry+10, 0, n/2*ry*.9, 0, 0]
    x0_l = np.array(base + [base[i]-2.5 for i in range(6)] + [-2.5 for _ in range(3)] + [0 for _ in range(6)])
    x0_u = np.array(base + [base[i]+2.5 for i in range(6)] + [2.5 for _ in range(3)] + [0 for _ in range(6)])
    # x0_l = np.array(base + base + [0,0])
    # x0_u = np.array(base + base + [0,0])
    scenario.set_init(
        [
            [x0_l.tolist(), 
             x0_u.tolist()],
        ],
        [
            (GOMode.Passive, POMode.Passive, MoveMode.NMT)
        ],
    )

    start = time.perf_counter()    
    # trace = scenario.verify(T, 1)
    trace = scenario.verify(T, 1) # fastest for T = 3000, ts = 0.2 was around 3.5 minutes
    

    print(f'Ver/sim time: {time.perf_counter()-start:.3f}')
    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 1, 2, [1,2], plot_color=colors)
    fig.data[0].name = 'True State'
    fig.data[0].showlegend = True

    # N = 20
    # sim_traces = []
    # for _ in range(N):
    #     sim_traces.append(scenario.simulate(T,1))
    # for st in sim_traces:
    #     fig = simulation_tree(st, None, fig, 1, 2, [1,2])

    # fig = reachtube_tree(trace, None, fig, 7, 8, [7,8])
    # fig.data[-1].name = 'Est State'
    # fig.data[-1].showlegend = True

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            x_sol, u_sol = pickle.load(f)
            u_sol = np.vstack([u_sol, u_sol[-1]])
        os.remove(filename)

    for f in files:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(cache_base):
        shutil.rmtree(cache_base)

    fig.add_trace(
        go.Scatter(
            x=x_sol[:, 0],
            y=x_sol[:,1],
            mode="lines",
            line_color="#000000",
            name="Reference Trajectory"
    ))

    fig.update_layout(
        xaxis_title='x (km)',
        yaxis_title='y (km)',
        legend_title='Trajectory Types',
    )
    # if os.path.exists(filename):
    #     with open(filename, 'rb') as f:
    #         x_sol, u_sol = pickle.load(f)
    #         u_sol = np.vstack([u_sol, u_sol[-1]])
    #     os.remove(filename)
    
    # fig = reachtube_tree(trace, None, fig, 0, 13)

    # ground = np.zeros((3001, 2, 6)) # time horizon + 1 / ts, 2, all 6 states
    # est = np.zeros((3001,2,6))
    # refs = np.zeros((3001, 6))
    # for node in trace.nodes:
    #     tr = node.trace['deputy']
    #     for i in range(0, len(tr), 2):
    #         t = int(tr[i][0])
    #         ground[t][0] = tr[i][1:7]
    #         ground[t][1] = tr[i+1][1:7]
    #         est[t][0] = tr[i][7:13]
    #         est[t][1] = tr[i+1][7:13]
    # for t in range(3001):
    #     refs[t] = OrbitalAgent.x_ref_fn(t, 10, x_sol, u_sol)
    
    # ref_err_low = ground[:,0] - refs
    # ref_err_high = ground[:,1] - refs
    # ref_est_low = est[:,0] - refs
    # ref_est_high = est[:,1] - refs

    # fig.add_trace(
    #     go.Scatter(
    #         x=np.linspace(0, 3001, 3000),
    #         y= ref_err_low[:,0],
    #         mode = 'lines',
    #         line_color='#000000',
    #         showlegend=False
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=np.linspace(0, 3001, 3000),
    #         y= ref_err_high[:,0],
    #         mode = 'lines',
    #         line_color='#000000',
    #         name='true-ref error'
    #         )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=np.linspace(0, 3001, 3000),
    #         y= ref_est_low[:,0],
    #         mode = 'lines',
    #         line_color='#0000CC',
    #         showlegend=False
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=np.linspace(0, 3001, 3000),
    #         y= ref_est_high[:,0],
    #         mode = 'lines',
    #         line_color='#0000CC',
    #         name='est-ref error'
    #         )
    # )

    fig.show()