# from orbital_all_agent import OrbitalAgent
from orbital_docking_agent_gpa import OrbitalAgent # can also change controller so if within a certain radius of the chief, just stop docking altogether
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
from orbital_docking_sensor_gpa_parser_wrapper_freq import OrbitalSensor
from parsed_wrap import clear_parse_cache
from verse.utils.star_diams import time_step_diameter_rect, sim_traces_to_dict_composed, sim_traces_to_diameters

import plotly.graph_objects as go
from enum import Enum, auto
import pickle
import time
import os 

n = 0.00438138 # constant equal to (\mu/r_e^3)^{-3}
filename = "demo/aprod/refs.pkl"

class GOMode(Enum):
    Passive = auto()
    Active = auto()

class POMode(Enum):
    Passive = auto()
    Active = auto()

class MoveMode(Enum):
    NMT = auto()
    Docking = auto()
    Docked = auto()

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
    if os.path.exists(filename):
        os.remove(filename)

    clear_parse_cache()

    # input_code_name = "./demo/aprod/orbital_docking_controller.py"
    input_code_name = "./demo/aprod/orbital_docking_controller_gpa_parser_freq.py"
    # input_code_name = "./demo/aprod/orbital_docking_controller_proxdyn.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC # still works even with base dryvr
    dep = OrbitalAgent("deputy", file_name=input_code_name)
    scenario.add_agent(dep)
    orbital_sensor = OrbitalSensor()
    scenario.set_sensor(orbital_sensor)
    # modify mode list input
    # base = [10,20,0,1,2,0]
    # T = 3000
    T = 3000 # note that promixity sensor is moving target due to MPC  
    ry = 75
    base = [0, ry+10, 0, n/2*ry*.9, 0, 0]
    x0_l = np.array(base + [base[i]-2.5 for i in range(6)] + [-2.5 for _ in range(3)] + [0 for _ in range(7)])
    x0_u = np.array(base + [base[i]+2.5 for i in range(6)] + [2.5 for _ in range(3)] + [0 for _ in range(7)])
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
    # trace = scenario.verify(T, 5)
    ts = 1
    trace = scenario.verify(T, ts) # fastest for T = 3000, ts = 0.2 was around 3.5 minutes
    

    print(f'Ver/sim time: {time.perf_counter()-start:.3f}')
    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 1, 2, [1,2], plot_color=colors)
    fig.data[0].name = 'True State'
    # fig.data[0].showlegend = True
    diam = time_step_diameter_rect(trace, T, ts)
    print(diam)
    diam_0, diam_f, diam_bar = 45, diam[-1], (sum(diam)+0.0)/len(diam) # NOTE: manually computing correct L1 diameter values
    print(f'F/I: {diam_f/diam_0:.5f}, A/I: {diam_bar/diam_0:.5f}\n raw final: {diam_f:.5f}, raw average: {diam_bar:.5f}, raw initial: {diam_0:.5f}')
    
    # N = 25
    # start_time = time.perf_counter()
    # sim_traces = []
    # for i in range(N):
    #     sim_traces.append(scenario.simulate(T, 1))
    #     if i != N-1 and os.path.exists(filename):
    #         os.remove(filename)

    # fig = go.Figure()
    # for st in sim_traces:
    #     fig = simulation_tree(st, None, fig, 1, 2, [1,2], 'lines', 'trace')

    # print(f'Runtime for {N} sims, T={T}, ts={1}: {time.perf_counter()-start_time:.2f}')
    # # sim_dict = sim_traces_to_dict_composed(sim_traces)
    # diam_0 = 45
    # diam_sim = sim_traces_to_diameters(sim_traces)
    # diam_f_sim, diam_bar_sim = diam_sim[-1], (sum(diam_sim)+0.0)/len(diam_sim)
    # print(f'Sim results: F/I: {diam_f_sim/diam_0:.5f}, A/I: {diam_bar_sim/diam_0:.5f}\n raw final: {diam_f_sim:.5f}, raw average: {diam_bar_sim:.5f}')


    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            x_sol, u_sol = pickle.load(f)
            u_sol = np.vstack([u_sol, u_sol[-1]])
        os.remove(filename)

    # fig.add_trace(
    #     go.Scatter(
    #         x=x_sol[:, 0],
    #         y=x_sol[:,1],
    #         mode="lines",
    #         line_color="#000000",
    #         name="Reference Trajectory"
    # ))

    # fig.update_layout(
    #     xaxis_title='x (km)',
    #     yaxis_title='y (km)',
    #     legend_title='Trajectory Types',
    # )

    fig.update_layout(
        width=1000,
        height=500,  # Starting with a 2:1 canvas ratio
        plot_bgcolor='white',
        margin=dict(l=150, r=50, b=120, t=50),
        
        # X-AXIS: Forced viewing window
        xaxis=dict(
            title='x (km)',
            range=[-10, 50],  # EXACT VIEWING WINDOW
            tickmode='array',
            tickvals=[-10, 0, 10, 20, 30, 40, 50],
            constrain='domain',
            title_font=dict(size=50, family='Arial, Bold', color='black'),
            tickfont=dict(size=45, family='Arial', color='black'),
            showline=True, linewidth=4, linecolor='black', mirror=True,
            ticks='outside', tickwidth=4, ticklen=18,
            showgrid=True, gridcolor='lightgray', griddash='dash',
            zeroline=False
        ),
        
        # Y-AXIS: Forced viewing window
        yaxis=dict(
            title='y (km)',
            range=[-100, 100], # EXACT VIEWING WINDOW
            tickmode='array',
            tickvals=[-100, -50, 0, 50, 100],
            title_font=dict(size=50, family='Arial, Bold', color='black'),
            tickfont=dict(size=45, family='Arial', color='black'),
            showline=True, linewidth=4, linecolor='black', mirror=True,
            ticks='outside', tickwidth=4, ticklen=18,
            title_standoff=30,
            showgrid=True, gridcolor='lightgray', griddash='dash',
            zeroline=False
        )
    )

    fig.show()