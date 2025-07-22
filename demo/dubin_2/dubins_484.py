from dubins_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse import Scenario, ScenarioConfig
from verse.analysis import Verifier
from verse.sensor import BaseSensor
from verse.analysis.verifier import ReachabilityMethod
import sys
import plotly.graph_objects as go
import plotly.io as pio
import torch
# from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
import pyvista as pv
import time
from dubin_sensor import DubinSensor
from verse.plotter.plotter3D import *

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from verse.stars import StarSet
from verse.sensor.base_sensor_stars import BaseStarSensor
from verse.plotter.plotterStar import plot_agent_trace
import polytope as pc
import multiprocessing
from datetime import date
import os
import pyvistaqt as pvqt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, 
    QSizePolicy, QLabel, QLineEdit, QTextEdit,QComboBox, QSlider
)

import imageio.v2 as imageio

class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()

class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
    M21 = auto()
    M10 = auto()

def tree_safe(tree: AnalysisTree):
    for node in tree.nodes:
        if node.assert_hits is not None:
            return False 
    return True

refine_profile = {
    'R1': [0],
    'R2': [0],
    'R3': [0,1,2,3]
}
def verify_refine(scenario : Scenario, time_horizon : float, time_step : float):
    assert time_horizon > 0
    assert time_step > 0

    own_acas_plane = scenario.init_dict['car1']
    int_npc_plane = scenario.init_dict['car2']
    partition_depth = 0
    if own_acas_plane[1][3] - own_acas_plane[0][3] > 0.05:
        exp = 'R3'
    elif own_acas_plane[1][0] - own_acas_plane[0][0] >= 75:
        exp = 'R2'
    else:
        exp = 'R1'
    res_list = []
    init_queue = []

    # x-coordinate
    if own_acas_plane[1][0]-own_acas_plane[0][0] > 1:
        car_x_init_range = np.linspace(own_acas_plane[0][0], own_acas_plane[1][0], 10)
    else:
        car_x_init_range = [own_acas_plane[0][0], own_acas_plane[1][0]]
    
    # y-coordinate
    if own_acas_plane[1][1]-own_acas_plane[0][1] > 1:
        car_y_init_range = np.linspace(own_acas_plane[0][1], own_acas_plane[1][1], 10)
    else:
        car_y_init_range = [own_acas_plane[0][1], own_acas_plane[1][1]]
    
    # theta-angle
    if own_acas_plane[1][2]-own_acas_plane[0][2] > np.pi/180:
        car_theta_init_range = np.linspace(own_acas_plane[0][2], own_acas_plane[1][2], 5)
    else:
        car_theta_init_range = [own_acas_plane[0][2], own_acas_plane[1][2]]

    # velocity
    if own_acas_plane[1][3]-own_acas_plane[0][3] > 0.1: 
        car_v_init_range = np.linspace(own_acas_plane[0][3], own_acas_plane[1][3], 5)
    else:
        car_v_init_range = [own_acas_plane[0][3], own_acas_plane[1][3]]

    for x in range(len(car_x_init_range) - 1):
        for y in range(len(car_y_init_range) - 1):
            for theta in range(len(car_theta_init_range) - 1):
                for velocity in range(len(car_v_init_range) - 1):
                    tmp = copy.deepcopy(own_acas_plane)

                    tmp[0][0] = car_x_init_range[x]
                    tmp[1][0] = car_x_init_range[x + 1]

                    tmp[0][1] = car_y_init_range[y]
                    tmp[1][1] = car_y_init_range[y + 1]

                    tmp[0][1] = car_theta_init_range[theta]
                    tmp[1][1] = car_theta_init_range[theta + 1]

                    tmp[0][3] = car_v_init_range[velocity]
                    tmp[1][3] = car_v_init_range[velocity + 1]

                    init_queue.append((tmp, int_npc_plane, partition_depth))

    start = len(init_queue)
    from alive_progress import alive_bar
    progress = 0
    with alive_bar(100, max_cols = 140, manual =True) as safe_bar:
        safe_bar.title('% Verified Safe (May be at 0% for some time)')

        while init_queue!=[]:
            own_plane, int_plane, partition_depth = init_queue.pop(0)
            if(partition_depth >=5):
                print("If the % Safe bar hasn't gone up past 0 by now, we recommend exiting")
            print(f"######## Current Partition Depth: {partition_depth}, own x: [{own_plane[0][0]}, {own_plane[1][0]}], own y : [{own_plane[0][1]}, {own_plane[1][1]}], own theta : [{own_plane[0][2]}, {own_plane[1][2]}],  own v: [{own_plane[0][3]}, {own_plane[1][3]}]")
            scenario.set_init_single('car1', own_plane, (AgentMode.COC,))
            scenario.set_init_single('car2', int_plane, (AgentMode.COC,))
            traces = scenario.verify(time_horizon, time_step)
            scenario.verifier.loop_cache = []

            if not tree_safe(traces):
                # Partition ownship plane and intruder plane initial state
                idx = refine_profile[exp][partition_depth%len(refine_profile[exp])]
                if own_plane[1][idx] - own_plane[0][idx] < 0.001:
                    print(f"Can't partition initial set dimension {idx} anymore. Scenario is likely unsafe😭")
                    res_list.append(traces)
                    return res_list
                car_v_init = (own_plane[0][idx] + own_plane[1][idx])/2
                own_plane1 = copy.deepcopy(own_plane)
                own_plane1[1][idx] = car_v_init 
                init_queue.append((own_plane1, int_plane, partition_depth+1))
                own_plane2 = copy.deepcopy(own_plane)
                own_plane2[0][idx] = car_v_init 
                init_queue.append((own_plane2, int_plane, partition_depth+1))

            else:
                progress +=(1/(start*(2**partition_depth))) 
                safe_bar(progress)

                res_list.append(traces)
            print("Verify Tasks left to run: " + str(len(init_queue)))
        print( "Verify Refine: Scenario is SAFE😃")
        return res_list
           
def run(name, initial_states, queue, num_sims):
    try:
        import plotly.io as pio
        pio.renderers.default = 'browser'  # Ensures plot opens in browser

        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "controller_v3.py")
        car = CarAgent('car1', file_name=input_code_name)
        car2 = NPCAgent('car2')

        scenario = Scenario(ScenarioConfig(parallel=False))
        scenario.set_sensor(DubinSensor())

        initial_own_plane = initial_states[0]
        initial_int_plane = initial_states[1]

        scenario.add_agent(car)
        scenario.add_agent(car2)

        scenario.set_init_single('car1', initial_own_plane, (AgentMode.COC,))
        scenario.set_init_single('car2', initial_int_plane, (AgentMode.COC,))

        start = time.perf_counter()
        trace = scenario.simulate(20, 0.1)
        fig2 = go.Figure()
        fig2 = simulation_tree(trace)
        runtime = time.perf_counter() - start

        name1 = trace.root.init
        queue1 = [trace.root]

        # Annotate trajectories with mode names
        while queue1:
            node = queue1.pop(0)
            traces = node.trace
            for agent_id in traces:
                trace_data = np.array(traces[agent_id])
                if len(trace_data) == 0:
                    continue

                x_vals = trace_data[:, 1]
                y_vals = trace_data[:, 2]
                mode = node.mode[agent_id][0]  # Example: "COC"

                for idx in range(0, len(trace_data), 20):  # label every 20th point
                    fig2.add_trace(
                        go.Scatter(
                            x=[x_vals[idx]],
                            y=[y_vals[idx]],
                            mode="text",
                            text=[f"{mode}"],
                            textposition="middle center",
                            textfont=dict(size=10),
                            showlegend=False,
                        )
                    )

            queue1 += node.child

        # Show the interactive figure in browser
        fig2.show()

        # Save logs and image if needed
        today = date.today()
        date_str = today.strftime("%Y-%m-%d")
        result_dir = os.path.join(script_dir, "results", f"{date_str}", f"{name}")
        os.makedirs(result_dir, exist_ok=True)
        output_dir = os.path.join(result_dir, f"simulation_{num_sims}")
        os.makedirs(output_dir, exist_ok=True)

        reach_path = os.path.join(output_dir, f"{num_sims}.png")
        fig2.write_image(reach_path, width=2000, height=800, scale=2)  # optional image save

        info_path = os.path.join(output_dir, "info.txt")
        with open(info_path, "a") as f:
            f.write(f"Name : {name}\n")
            f.write(f"Verification_runtime: {runtime:.3f}s\n")
            f.write("Initial sets are: \n")
            f.write(f"   car1 : {initial_states[0]}\n")
            f.write(f"   car2 : {initial_states[1]}\n")
            f.write(f"   initial_car1 : {name1['car1']}\n")
            f.write(f"   initial_car2 : {name1['car2']}\n")

        print("[Child] Finished verification. Runtime:", runtime)
        queue.put(runtime)
        print("[Child] Done putting result", flush=True)
        trace = None
    except Exception as e:
        print("[Child] Exception occurred:", e)
        queue.put(None)


if __name__ == "__main__":
    
    # pio.renderers.default = 'browser'
    # initial_states =  [[[-4000, -1000, -np.pi/2, 200, 0], [-3000, 1000, np.pi/2, 200, 0]], [[-1000, -5000, -np.pi/2, 400, 0], [1000, 5000, np.pi/2, 400, 0]]] 
    num_sims = 4
    # output_gif_path: str = "simulation_trajectories.gif",
    frames = []

    script_dir = os.path.realpath(os.path.dirname(__file__))

    # input_code_name = os.path.join(script_dir, "controller_v3.py")
    # car = CarAgent('car1', file_name=input_code_name)
    # car2 = NPCAgent('car2')
    # scenario = Scenario(ScenarioConfig(parallel=False))
    # scenario.set_sensor(DubinSensor())


    # # initial_states = [[[0, 0, -np.pi, 100, 0], [30000, 30000, np.pi, 500, 0]], [[0, 0, -np.pi, 0, 0], [0, 0, np.pi, 500, 0]]]
    # # initial_states = [[[-30000, -30000, -np.pi, 100, 0], [30000, 30000, np.pi, 500, 0]], [[0, 0, -np.pi, 0, 0], [40000, 40000, np.pi, 500, 0]]] 


    # initial_own_plane = initial_states[0]
    # initial_int_plane = initial_states[1]

    # scenario.add_agent(car)
    # scenario.add_agent(car2)

    # scenario.set_init_single('car1', initial_own_plane, (AgentMode.COC,))
    # scenario.set_init_single('car2', initial_int_plane, (AgentMode.COC,))
    # plotter = go.Figure()
    # start = time.perf_counter()
    # # scenario.config.reachability_method = ReachabilityMethod.DRYVR
    # # plotter = pvqt.QtInteractor()
    # # plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # fig2 = go.Figure()
    # agent_id = 'car1'
    # trace = scenario.simulate(20, 0.1)
    # fig2 = simulation_tree(trace)

    # runtime = time.perf_counter() - start
    # # plotRemaining(plotter, False)
    # print(trace.root.init)


    # # fig1 = reachtube_tree(trace, fig1, x_dim = 1, x_title = 'x', y_dim = 2, y_title = 'y', z_dim = 0, z_title = 'time')
    # # fig1.show()

    # queue = [trace.root]
    # # print(trace.root[:, 1], trace.root[:, ])
    # while queue:
    #     node = queue.pop(0)
    #     traces = node.trace
    #     for agent_id in traces:
    #         trace = np.array(traces[agent_id])
    #         if len(trace) == 0:
    #             continue

    #         # Extract x, y coordinates and mode
    #         x_vals = trace[:, 1]
    #         y_vals = trace[:, 2]
    #         mode = node.mode[agent_id][0]  # Assuming mode is a tuple with AgentMode as first element

    #         # Add annotations for every nth point
    #         for idx in range(0, len(trace), 20):
    #             fig2.add_trace(
    #                 go.Scatter(
    #                     x=[x_vals[idx] * 11/ 10 if x_vals[idx] < 1000 else x_vals[idx] + 50],
    #                     y=[y_vals[idx] * 11/ 10 if y_vals[idx] < 1000 else y_vals[idx] + 50],
    #                     mode="text",
    #                     text=[f"{mode}"],
    #                     textposition= "middle center",
    #                     textfont=dict(size=text_size, color= None),
    #                     showlegend=False,
    #                 )
    #             )

    #     queue += node.child
    #     # temp_image_path = f"temp_frame_{i}.png"
    #     # fig2.write_image(temp_image_path, width=800, height=600, scale=2)

    #     # # Read image and append to frames
    #     # frames.append(imageio.imread(temp_image_path))

    #     # # Clean up temporary file
    #     # os.remove(temp_image_path)

    # fig2.show()
    # print(runtime)
   
    list_of_initial_states = {
        # "safjdsl"        : [[[-4000, -1000, -np.pi/2, 200, 0], [-3000, 1000, np.pi/2, 100, 0]], [[-1000, -5000, -np.pi/2, 100, 0], [1000, 5000, np.pi/2, 400, 0]]] ,
        # "initials_condition_2"        : [[[-2000, -4000, -np.pi/2, 200, 0], [-2000, 2000, np.pi/2, 200, 0]], [[-2000, -5000, -np.pi/2, 200, 0], [2000, 5000, np.pi/2, 400, 0]]],
        # "initials_condition_3"        : [[[-4000, -4000, -np.pi/2, 100, 0], [-3000, 2000, np.pi/2, 200, 0]], [[-3000, -5000, -np.pi/2, 300, 0], [-1000, 5000, np.pi/2, 400, 0]]],
        # "initials_condition_4"        : [[[-4000, -4000, -np.pi/2, 300, 0], [-3000, 2000, np.pi/2, 500, 0]], [[-3000, -5000, -np.pi/2, 400, 0], [-1000, 5000, np.pi/2, 400, 0]]],
        # "initials_condition_5"        : [[[-7000, -7000, -np.pi/2, 0, 0], [-6000, 4000, np.pi/2, 200, 0]], [[-3000, -5000, -np.pi/2, 200, 0], [-1000, 5000, np.pi/2, 400, 0]]],
    }
    for name, initial_states in list_of_initial_states.items():
        print(f"Running {name}, initial_states : {initial_states}")
        queue = multiprocessing.Queue()

        p = multiprocessing.Process(target= run, args=(name, initial_states, queue, num_sims))
        p.start()
        p.join(timeout=7200)
        today = date.today()
        date_str = today.strftime("%Y-%m-%d")  
        result_dir = os.path.join(script_dir, "results", f"{date_str}")
        log_path = os.path.join(result_dir, "log.txt")   
        if p.is_alive():
            with open(log_path, "w") as f:
                f.write(f"Name : {name}\n")
                f.write("Initial sets are: \n")
                f.write(f"   car1 : {initial_states[0]}\n")
                f.write(f"   car2 : {initial_states[1]}\n")
                f.write("FAILED: This initial_state is taking more than 2 hours \n \n")
            p.terminate()
            p.join()
            result = None
        else:
            result = queue.get() if not queue.empty() else None
            with open(log_path, "w") as f:
                f.write(f"Name : {name}\n")
                f.write("Initial sets are: \n")
                f.write(f"   car1 : {initial_states[0]}\n")
                f.write(f"   car2 : {initial_states[1]}\n")
                f.write("SUCCEED: This initial_state has finished in required time\n\n")
