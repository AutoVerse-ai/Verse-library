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
from verse.plotter.plotter3D_new import *

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from verse.stars import StarSet
from verse.sensor.base_sensor_stars import BaseStarSensor
from verse.plotter.plotterStar import plot_agent_trace
import polytope as pc
import multiprocessing
from datetime import date
import os

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




def plot_stars(stars: List[StarSet], dim1: int = None, dim2: int = None):
    for star in stars:
        x, y = np.array(star.get_verts(dim1, dim2))
        plt.plot(x, y, lw = 1)
        centerx, centery = star.get_center_pt(0, 1)
        plt.plot(centerx, centery, 'o')
    plt.show()

def verify_all(scenario : Scenario, time_horizon : float, time_step : float):
    assert time_horizon >= 0
    assert time_step > 0
    verification_methods = ["DRYVR", "NEU_REACH", "MIXMONO_CONT", "MIXMONO_DISC", "DRYVR_DISC", "STAR_SETS"]
    sensors = ["BaseSense", "DubinsSense"]

    script_dir = os.path.realpath(os.path.dirname(__file__))

    for method in verification_methods:
        for sensor in sensors:
            if verification_methods == "DRYVR":
                scenario.verifier = Verifier(config=ScenarioConfig(ReachabilityMethod.DRYVR))
            if verification_methods == "NEU_REACH":
                scenario.verifier = Verifier(config=ScenarioConfig(ReachabilityMethod.NEU_REACH))
            if verification_methods == "MIXMONO_CONT":
                scenario.verifier = Verifier(config=ScenarioConfig(ReachabilityMethod.MIXMONO_CONT))
            if verification_methods == "MIXMONO_DISC":
                scenario.verifier = Verifier(config=ScenarioConfig(ReachabilityMethod.MIXMONO_DISC))
            if verification_methods == "DRYVR_DISC":
                scenario.verifier = Verifier(config=ScenarioConfig(ReachabilityMethod.DRYVR_DISC))
            if verification_methods == "STAR_SETS":
                scenario.verifier = Verifier(config=ScenarioConfig(ReachabilityMethod.STAR_SETS))
            
            if sensor == "BaseSense":
                scenario.set_sensor(BaseSensor())
            if sensor == "DubinsSense":
                scenario.set_sensor(DubinSensor())
            
            result_dir = os.path.join(script_dir, "results", f"{sensor}_{method}")
            os.makedirs(result_dir, exist_ok=True)
            info_path = os.path.join(result_dir, "info.txt")
            trace = None

            try:
                start = time.perf_counter()
                print (f"Verifting with {sensor} sensor and {method} verification method")
                trace = scenario.verify(time_horizon=time_horizon, time_step=time_step)

                verification_time = time.perf_counter()-start

                
                with open(info_path, "w") as f:
                    f.write(f"Verification_runtime: {verification_time:.3f}s\n")
                    
                
                fig1 = simulation_tree(trace)
                
                
                fig2 = reachtube_tree(trace)

                # print(type(fig1))

                # Testing 
                # fig1 = Figure()
                # canvas = FigureCanvas(fig1)  # Attach canvas to figure

                # # Add a subplot and plot data
                # ax = fig1.add_subplot(111)
                # ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
                # ax.set_title("Sample Plot")

                # fig2 = Figure()
                # canvas = FigureCanvas(fig2)  # Attach canvas to figure

                # # Add a subplot and plot data
                # ax = fig2.add_subplot(111)
                # ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
                # ax.set_title("Sample Plot")
                fig1.update_layout(
                    title="Simulation Results",
                    xaxis_title="Time (s)",
                    yaxis_title="Position (m)",
                )

                fig2.update_layout(
                    title="Reachable Set",
                    xaxis_title="X Coordinate (m)",
                    yaxis_title="Y Coordinate (m)",
                )
                sim_path = os.path.join(result_dir, "simulation.png")
                fig1.write_image(sim_path, width=2000, height=800, scale = 2)
                # fig1.show()

                # Save fig2
                reach_path = os.path.join(result_dir, "reachable.png")
                fig2.write_image(reach_path, width=2000, height=800, scale = 2)
                # fig1.show()



            except Exception as e:
                with open(info_path, "w") as f:
                    f.write(f"Errors happened in sensor = {sensor} and method = {method}", e)

refine_profile = {
    'R1': [0],
    'R2': [0],
    'R3': [0,0,0,3]
}
def verify_refine(scenario : Scenario, time_horizon : float, time_step : float):
    assert time_horizon > 0
    assert time_step > 0

    own_acas_plane = scenario.init_dict['car1']
    int_npc_plane = scenario.init_dict['car2']
    partition_depth = 0
    if own_acas_plane[1][3] - own_acas_plane[0][3]>0.1:
        exp = 'R3'
    elif own_acas_plane[1][0] - own_acas_plane[0][0] >= 75:
        exp = 'R2'
    else:
        exp = 'R1'
    res_list = []
    init_queue = []

    # x-coordinate
    if own_acas_plane[1][0]-own_acas_plane[0][0] > 1:
        car_x_init_range = np.linspace(own_acas_plane[0][0], own_acas_plane[1][0], 5)
    else:
        car_x_init_range = [own_acas_plane[0][0], own_acas_plane[1][0]]
    
    # y-coordinate
    if own_acas_plane[1][1]-own_acas_plane[0][1] > 1:
        car_y_init_range = np.linspace(own_acas_plane[0][1], own_acas_plane[1][1], 5)
    else:
        car_y_init_range = [own_acas_plane[0][1], own_acas_plane[1][1]]
    
    # theta-angle
    if own_acas_plane[1][2]-own_acas_plane[0][2] > 0.01:
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
           
def run(name, initial_states, queue):
    try:
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
        trace = scenario.verify(20, 0.1)
        runtime = time.perf_counter() - start

        fig1 = reachtube_tree(trace)
        fig1.show()

        fig2 = simulation_tree(trace)
        fig2.show()

        fig1.update_layout(
                title="Simulation Results",
                xaxis_title="X Coordinate (m)",
                yaxis_title="Y Coordinate (m)",
            )

        fig2.update_layout(
            title="Reachable Set Results",
            xaxis_title="X Coordinate (m)",
            yaxis_title="Y Coordinate (m)",
        )
        today = date.today()
        date_str = today.strftime("%Y-%m-%d")  
        result_dir = os.path.join(script_dir, "results", f"{date_str}")
        os.makedirs(result_dir, exist_ok=True)
        output_dir = os.path.join(result_dir, f"{name}")
        os.makedirs(output_dir, exist_ok=True)
        info_path = os.path.join(output_dir, "info.txt")   

        sim_path = os.path.join(output_dir, "simulation.png")
        fig1.write_image(sim_path, width=2000, height=800, scale = 2)
            # fig1.show()

            # Save fig2
        reach_path = os.path.join(output_dir, "reachability.png")
        fig2.write_image(reach_path, width=2000, height=800, scale = 2)

        with open(info_path, "w") as f:
            f.write(f"Name : {name}\n")
            f.write(f"Verification_runtime: {runtime:.3f}s\n")
            f.write("Initial sets are: \n")
            f.write(f"   car1 : {initial_states[0]}\n")
            f.write(f"   car2 : {initial_states[1]}\n")

        print("[Child] Finished verification. Runtime:", runtime)
        queue.put(runtime)
        print("[Child] Done putting result", flush=True)
    except Exception as e:
        print("[Child] Exception occurred:", e)
        queue.put(None)


if __name__ == "__main__":
    
    
    pio.renderers.default = 'browser'

    script_dir = os.path.realpath(os.path.dirname(__file__))
    # input_code_name = os.path.join(script_dir, "controller_v3.py") # Contains ACAS Xu Decision Logic
    # car = CarAgent('car1', file_name=input_code_name)
    # car2 = NPCAgent('car2')
    # scenario = Scenario(ScenarioConfig(parallel=False))
    # scenario.set_sensor(DubinSensor())

    # Verification time : 11299.587s (16758 nodes generated)
    # initial_own_plane = [[0, 0, 0, 200, 0], [0, 0, np.pi/4, 200, 0]]
    # initial_int_plane = [[0, 10000, -np.pi/2, 400, 0], [0, 10000, -np.pi/2, 400, 0]]
    
    list_of_initial_states = {
        "one_CL_3_y"        : [[[-1000, 0, np.pi/2, 200, 0], [0, 100, np.pi/2, 200, 0]], [[5000, 0, np.pi/2 + 0.4296, 400, 0], [5000, 0, np.pi/2 + 0.4296, 400, 0]]],
        "one_CL_1_y"        : [[[0, -100, np.pi/2, 200,  0], [0, 100, np.pi/2, 200, 0]], [[-5000, 0, np.pi/2 - 0.4296, 400, 0], [-5000, 0, np.pi/2 - 0.4296, 400, 0]]]
    }
    # "single_CL_2"       : [[[0, 0, np.pi/2, 200, 0], [0, 0, np.pi/2, 200, 0]], [[0, -5000, np.pi/2, 400, 0], [0, -5000, np.pi/2, 400, 0]]], 
    # "one_CL_2_x"        : [[[-100, 0, np.pi/2, 200, 0], [100, 0, np.pi/2, 200, 0]], [[0, -5000, np.pi/2, 400, 0], [0, -5000, np.pi/2, 400, 0]]],
    #  "one_CL_2_y"       : [[[0, -100, np.pi/2, 200, 0], [0, 100, np.pi/2, 200, 0]], [[0, -5000, np.pi/2, 400, 0], [0, -5000, np.pi/2, 400, 0]]],
    # "one_CL_2_theta"    : [[[0, 0, np.pi/2 - 0.1, 200, 0], [0, 0, np.pi/2 + 0.1, 200, 0]], [[0, -5000, np.pi/2, 400, 0], [0, -5000, np.pi/2, 400, 0]]],
    for name, initial_states in list_of_initial_states.items():
        print(f"Running {name}, initial_states : {initial_states}")
        queue = multiprocessing.Queue()

        p = multiprocessing.Process(target= run, args=(name, initial_states, queue))
        p.start()
        p.join(timeout=7200)
        today = date.today()
        date_str = today.strftime("%Y-%m-%d")  
        result_dir = os.path.join(script_dir, "results", f"{date_str}")
        log_path = os.path.join(result_dir, "log.txt")   
        if p.is_alive():
            with open(log_path, "a") as f:
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
            with open(log_path, "a") as f:
                f.write(f"Name : {name}\n")
                f.write("Initial sets are: \n")
                f.write(f"   car1 : {initial_states[0]}\n")
                f.write(f"   car2 : {initial_states[1]}\n")
                f.write("SUCCEED: This initial_state has finished in required time\n\n")



    # scenario.add_agent(car)
    # scenario.add_agent(car2)

    # scenario.set_init_single('car1', initial_own_plane, (AgentMode.COC,))
    # scenario.set_init_single('car2', initial_int_plane, (AgentMode.COC,))

    # start = time.perf_counter()

    # trace = None
    
    # try:
    #     start = time.perf_counter()
        
    #     trace = scenario.verify(20, 0.1)
    #     print(f'Verification runtime: {time.perf_counter()-start:.3f}s')

    #     fig2 = reachtube_tree(trace)
    #     fig2.show()

    #     fig4 = simulation_tree(trace)
    #     fig4.show()

    # except KeyboardInterrupt:
    #     elapsed = time.perf_counter() - start
    #     print(f"\nKeyboardInterrupt: Program stopped after {elapsed:.3f} seconds.")
    #     if trace is not None:
    #         print(f"Partial trace was collected. Number of nodes: {len(trace.nodes)}")
    #     else:
    #         print("No trace was collected yet.")