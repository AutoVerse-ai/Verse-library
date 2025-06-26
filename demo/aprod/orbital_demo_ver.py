from orbital_agent import OrbitalAgent, OpenOrbitalAgent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *

import plotly.graph_objects as go
from enum import Enum, auto
import pickle
import time

n = 0.00438138 # constant equal to (\mu/r_e^3)^{-3}

class OrbitalMode(Enum):
    Passive = auto()

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
    input_code_name = "./demo/aprod/orbital_controller.py"
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    dep = OrbitalAgent("deputy", file_name=input_code_name)
    scenario.add_agent(dep)
    # scenario.set_sensor(CraftSensor())
    # modify mode list input
    base = [10,20,0,1,2,0]
    x0_l = np.array(base + [base[i]-2 for i in range(3)] + base[3:])
    x0_u = np.array(base + [base[i]+2 for i in range(3)] + base[3:])
    scenario.set_init(
        [
            [x0_l, 
             x0_u],
        ],
        [
            (OrbitalMode.Passive,)
        ],
    )

    start = time.perf_counter()    
    trace = scenario.verify(2000, 1)
    print(f'Simulaion time: {time.perf_counter()-start:.3f}')
    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 1, 2, [1,2], plot_color=colors)
    fig.data[0].name = 'True State'
    fig.data[0].showlegend = True

    fig = reachtube_tree(trace, None, fig, 7, 8, [7,8])
    fig.update_layout(
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        legend_title='Trajectory Types',
    )
    fig.data[-1].name = 'Est State'
    fig.data[-1].showlegend = True
    fig.show()