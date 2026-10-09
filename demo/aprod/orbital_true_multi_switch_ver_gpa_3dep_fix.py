# from orbital_all_agent import OrbitalAgent
from orbital_true_multi_switch_agent_gpa_3dep import OrbitalAgent, SimpleTrackingOrbitalAgent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
from orbital_true_multi_switch_sensor_gpa_fix import OrbitalSensor
from verse.utils.star_diams import time_step_diameter_rect, sim_traces_to_dict_composed, sim_traces_to_diameters

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
filename_obs = "demo/aprod/refs_obs.pkl"
filename_obs_inner = "demo/aprod/refs_obs_ra.pkl"
filename_aheader = "demo/aprod/refs_aheader.pkl"
filename_aheader_inner = "demo/aprod/refs_aheader_ra.pkl"

filenames = [filename, filename_ra, filename_ahead, filename_ahead_ra, filename_aheader, filename_aheader_inner]

class GOMode(Enum):
    Passive = auto()
    Active = auto()

class POMode(Enum):
    Passive = auto()
    Active = auto() # in this instance, there is no obstacle so only activity should be from being close to chief

class POTwoMode(Enum):
    Passive = auto()
    Active = auto

class PriorityMode(Enum):
    First = auto()
    Second = auto()
    Third = auto()

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

    input_code_name = "./demo/aprod/orbital_true_multi_switch_controller_gpa_fix.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    dep = OrbitalAgent("deputy", file_name=input_code_name)
    dep2 = OrbitalAgent('deputy_ahead', file_name=input_code_name)
    deper = OrbitalAgent('deputy_aheader', file_name=input_code_name)
    scenario.add_agent(dep)
    scenario.add_agent(dep2)
    scenario.add_agent(deper)
    orbital_sensor = OrbitalSensor()
    scenario.set_sensor(orbital_sensor)
    # modify mode list input
    T = 5500 # 3000 for about half a cycle and around 5500 for a full cycle
    ry = 75
    r_inner = ry - 20 
    x0_nmt = np.array([0, ry, 0, n/2*ry, 0, 0])
    x0_nmt_aheader = np.array([37.51233, -0.111  ,  0.     , -0.00007, -0.09003,  0.     ])

    base = [0, ry+10, 0, n/2*ry*.9, 0, 0]
    x0_l = np.array(base + [base[i]-2.5 for i in range(6)] + [-2.5 for _ in range(3)] + [0 for _ in range(8)])
    x0_u = np.array(base + [base[i]+2.5 for i in range(6)] + [2.5 for _ in range(3)] + [0 for _ in range(8)])
    
    base_ahead = [ 8.91431-5, 72.85074+5,  0.     ,  0.04371, -0.02139,  0.     ]
    x0_l_ahead = np.array(base_ahead + [base_ahead[i]-2.5 for i in range(3)] +[0 for _ in range(3)]+ [-2.5 for _ in range(3)] + [0 for _ in range(3)] + [1, 0, 0, 0, 1]) # desynchronizing the timers
    x0_u_ahead = np.array(base_ahead + [base_ahead[i]+2.5 for i in range(3)] +[0 for _ in range(3)] + [2.5 for _ in range(3)] +  [0 for _ in range(3)] + [1, 0, 0, 0, 1])
#   ahead should start by tracking: array([ 4.48926, 74.46068,  0.     ,  0.04468, -0.01077,  0.     ])

    x0_l_aheader = np.array(x0_nmt_aheader.tolist() + [x0_nmt_aheader[i]-2.5 for i in range(3)] +[0 for _ in range(3)]+ [-2.5 for _ in range(3)] + [0 for _ in range(3)] + [2, 0, 0, 0, 2]) # desynchronizing the timers
    x0_u_aheader = np.array(x0_nmt_aheader.tolist() + [x0_nmt_aheader[i]+2.5 for i in range(3)] +[0 for _ in range(3)] + [2.5 for _ in range(3)] +  [0 for _ in range(3)] + [2, 0, 0, 0, 2])


    scenario.set_init(
        [
            [x0_l.tolist(), 
             x0_u.tolist()],
            [x0_l_ahead.tolist(), x0_u_ahead.tolist()],
            [x0_l_aheader.tolist(), x0_u_aheader.tolist()],
        ],
        [
            # assign each agent an addition mode and state to denote whether an update occurred and priority resp.
            # actually just slightly stagger the timers 
            (GOMode.Passive, POMode.Passive, POTwoMode.Passive, PriorityMode.First, MoveMode.NMT),
            (GOMode.Passive, POMode.Passive, POTwoMode.Passive, PriorityMode.Second, MoveMode.NMT),
            # (MoveMode.Inner,),
            (GOMode.Passive, POMode.Passive, POTwoMode.Passive, PriorityMode.Third, MoveMode.NMT),
        ],
    )

    # start = time.perf_counter()    
    # trace = scenario.verify(T, 1)
    # print(f'Simulation time: {time.perf_counter()-start:.3f}')
    # fig = go.Figure()
    # fig = reachtube_tree(trace, None, fig, 1, 2, [1,2], plot_color=colors)
    # fig.data[0].name = 'True State'
    # fig.data[0].showlegend = True
    # diam = time_step_diameter_rect(trace, T, 1)
    # diam_0, diam_f, diam_bar = 105, diam[-1], (sum(diam)+0.0)/len(diam) # NOTE: manually computing correct L1 diameter values
    # print(f'F/I: {diam_f/diam_0:.5f}, A/I: {diam_bar/diam_0:.5f}\n raw final: {diam_f:.5f}, raw average: {diam_bar:.5f}, raw initial: {diam_0:.5f}')
    # fig = reachtube_tree(trace, None, fig, 7, 8, [7,8])


    N = 5
    start_time = time.perf_counter()
    sim_traces = []
    for i in range(N):
        sim_traces.append(scenario.simulate(T, 1))
        if i != N-1 and os.path.exists(f):
            for f in filenames:
                os.remove(f)

    fig = go.Figure()
    for st in sim_traces:
        fig = simulation_tree(st, None, fig, 1, 2, [1,2], 'lines', 'trace')

    print(f'Runtime for {N} sims, T={T}, ts={1}: {time.perf_counter()-start_time:.2f}')
    # sim_dict = sim_traces_to_dict_composed(sim_traces)
    diam_0 = 105
    diam_sim = sim_traces_to_diameters(sim_traces)
    diam_f_sim, diam_bar_sim = diam_sim[-1], (sum(diam_sim)+0.0)/len(diam_sim)
    print(f'Sim results: F/I: {diam_f_sim/diam_0:.5f}, A/I: {diam_bar_sim/diam_0:.5f}\n raw final: {diam_f_sim:.5f}, raw average: {diam_bar_sim:.5f}')

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