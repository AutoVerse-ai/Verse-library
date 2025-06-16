from satellite_agent import SatelliteAgent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *

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
    #   ['#6600CC', '#7F00FF', '#9933FF', '#B266FF', '#CC99FF', '#E5CCFF'], # purple
    ["#00CC00", "#00FF00", "#33FF33", "#66FF66", "#99FF99", "#E5FFCC"],  # lime
    # ["#CC0066", "#FF007F", "#FF3399", "#FF66B2", "#FF99CC", "#FFCCE5"],  # pink
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

def get_trace(trace: AnalysisTree) -> np.ndarray:
    return np.array(trace.root.trace['deputy'])

if __name__ == "__main__":
    input_code_name = "./demo/aprod/satellite_controller.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    dep = SatelliteAgent("deputy", file_name=input_code_name)
    scenario.add_agent(dep)
    true_dim_0 = 1
    est_dim_0 = 7

    true_dim = 2
    est_dim = 8
    # scenario.set_sensor(CraftSensor())
    # modify mode list input
    scenario.set_init(
        [
            # [[-925, -425, 1, 0, 0, 0, -925, -425, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
            #  [-875, -375, 1, 0, 0, 0, -875, -375, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
            # [[-900, -400, 0, 0, 0, 0, -925, -425, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
            #  [-900, -400, 0, 0, 0, 0, -875, -375, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
            [[0, -400, 0, 400*n, 0, 0, -925, -425, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0.05], 
             [0, -400, 0, 400*n, 0, 0, -875, -375, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0.05]],
        ],
        [
            (SatelliteMode.Passive,)
        ],
    )

    # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    # trace = scenario.verify(200, 0.1)
    # fig = go.Figure()
    # fig = reachtube_tree(trace, None, fig, 0, 17, plot_color=None)
    # fig = reachtube_tree(trace, None, fig, 0, 13, plot_color=colors)


    N = 1
    start = time.perf_counter()
    traces = []
    sim_traces = []
    for i in range(N):
        trace = scenario.simulate(200, 0.1)
        traces.append(trace)
        sim_traces.append(get_trace(trace))

    sim_traces = np.array(sim_traces)
    fig = go.Figure()
    i = 0
    
    # open("points.csv", "w").close()

    for trace in traces:

        

        # sim_trace = get_trace(trace)
        # true_rot_vects = quat_to_rot_vect(sim_trace[:,13:17])
        # h_rot_vects = quat_to_rot_vect(sim_trace[:,17:21])
        
        # fig.add_trace(
        #     go.Scatter(
        #         x=sim_trace[:, 0],
        #         y=h_rot_vects[:,0],
        #         mode="lines",
        #         line_color="#0000CC",
        #         showlegend=False,
        #     )
        # )
        if i>=5:
            continue
        # fig: go.Figure = simulation_tree(trace, None, fig, true_dim_0, true_dim, [true_dim_0, true_dim], "lines", "trace", plot_color=[['#000000']])
        # fig: go.Figure = simulation_tree(trace, None, fig, est_dim_0, est_dim, [est_dim_0, est_dim], "lines", "trace")
        fig: go.Figure = simulation_tree(trace, None, fig, 0, true_dim, [true_dim_0, true_dim], "lines", "trace", plot_color=[['#000000']])
        fig: go.Figure = simulation_tree(trace, None, fig, 0, est_dim, [est_dim_0, est_dim], "lines", "trace")
        i+=1

    fig.data[-2].name = 'Ground truth' # this only m
    fig.data[-1].name = 'Sampled traces'
    fig.data[-2].showlegend = True
    fig.data[-1].showlegend = True

    sample_min = np.min(sim_traces, 0)
    sample_max = np.max(sim_traces, 0)

    # sensed = np.loadtxt('points.csv', delimiter=',')
    # fig.add_trace(
    #     go.Scatter(
    #         x = sensed[:,0],
    #         y = sensed[:,1],
    #         mode = 'markers',
    #         line_color="#0000CC",
    #         name='Sensed data'
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         # x=sim_traces[0, :, 0],
    #         x=sample_min[:,est_dim_0],
    #         y=sample_min[:,est_dim],
    #         # y=sample_min[:,7],
    #         # mode="lines",
    #         mode="markers",
    #         line_color="#0000CC",
    #         # showlegend=False,
    #         text='Sample sensor min',
    #         name='Sample sensor min'
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         # x=sim_traces[0, :, 0],
    #         x=sample_max[:,est_dim_0],
    #         y=sample_max[:,est_dim],
    #         # y=sample_max[:,7],
    #         # mode="lines",
    #         mode="markers",
    #         line_color="#0000CC",
    #         # showlegend=False,
    #         text='Sample sensor max',
    #         name='Sample sensor max'
    #     )
    # )

    fig.update_layout(
        legend_title_text="Trace types",
        xaxis_title="time",
        yaxis_title="y"
    )
    fig.show()
    print(f'Total time for {N} simulations: {(time.perf_counter()-start)/60:.2f}')