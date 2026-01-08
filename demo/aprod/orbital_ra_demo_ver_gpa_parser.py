# from orbital_all_agent import OrbitalAgent
from orbital_ra_agent_gpa import OrbitalAgent, SimpleOrbtialAgent # at some point make all the different agents their own class
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *
from orbital_ra_sensor_gpa_parser import OrbitalSensor
from verse.utils.star_diams import time_step_diameter_rect, sim_traces_to_dict_composed, sim_traces_to_diameters

import plotly.graph_objects as go
from enum import Enum, auto
import pickle
import time
import os 

n = 0.00438138 # constant equal to (\mu/r_e^3)^{-3}
filename = "demo/aprod/refs.pkl"
filename_ra = "demo/aprod/refs_ra.pkl"
filenames = [filename, filename_ra]

class GOMode(Enum):
    Passive = auto()
    Active = auto()

class POMode(Enum):
    Passive = auto()
    OActive = auto()
    CActive = auto()
    OCActive = auto()

class TrajMode(Enum):
    Normal = auto()
    Avoid = auto()

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

    # input_code_name = "./demo/aprod/orbital_ra_controller_v2.py"
    input_code_name = "./demo/aprod/orbital_ra_controller_gpa.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    dep = OrbitalAgent("deputy", file_name=input_code_name)
    obs = SimpleOrbtialAgent("obs")
    scenario.add_agent(dep)
    scenario.add_agent(obs)
    orbital_sensor = OrbitalSensor()
    scenario.set_sensor(orbital_sensor)
    # modify mode list input
    # base = [10,20,0,1,2,0]
    T = 3000
    ry = 75
    base = [0, ry+10, 0, n/2*ry*.9, 0, 0]
    x0_l = np.array(base + [base[i]-2.5 for i in range(6)] + [-2.5 for _ in range(3)] + [0 for _ in range(9)])
    x0_u = np.array(base + [base[i]+2.5 for i in range(6)] + [2.5 for _ in range(3)] + [0 for _ in range(9)])
    obs_base = [0, -ry, 0, 0, 0, 0]

    scenario.set_init(
        [
            [x0_l.tolist(), 
             x0_u.tolist()],
             [obs_base, obs_base]
        ],
        [
            (GOMode.Passive, POMode.Passive, TrajMode.Normal),
            (GOMode.Passive,)
            # (OrbitalMode.Passive,)
        ],
    )

    start = time.perf_counter()    
    trace = scenario.verify(T, 1)
    print(f'Simulaion time: {time.perf_counter()-start:.3f}')

    diam = time_step_diameter_rect(trace, T, 1)
    diam_0, diam_f, diam_bar = 45, diam[-1], (sum(diam)+0.0)/len(diam) # NOTE: use correct diameter values
    print(f'F/I: {diam_f/diam_0:.5f}, A/I: {diam_bar/diam_0:.5f}\n raw final: {diam_f:.5f}, raw average: {diam_bar:.5f}, raw initial: {diam_0:.5f}')
    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 1, 2, [1, 2], "lines", "trace")

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