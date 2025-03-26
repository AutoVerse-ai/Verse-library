from wsgiref.validate import PartialIteratorWrapper
from mp0_p1 import VehicleAgent, PedestrianAgent, VehiclePedestrianSensor, VehiclePedestrianSensorStar, eval_velocity, sample_init
from verse import Scenario, ScenarioConfig
from vehicle_controller import VehicleMode, PedestrianMode

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *

from verse.stars.starset import *
from verse.sensor.base_sensor_stars import *
from verse.analysis.verifier import ReachabilityMethod
from mp0_p1 import get_extreme
import plotly.graph_objects as go
import copy

if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "vehicle_controller.py")
    vehicle = VehicleAgent('car', file_name=input_code_name)
    pedestrian = PedestrianAgent('pedestrian')

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS

    # scenario.add_agent(vehicle) 
    # scenario.add_agent(pedestrian)
    scenario.set_sensor(VehiclePedestrianSensorStar())

    # # ----------- Different initial ranges -------------
    '''This is one scenario where stars are safe but rectangles arent'''
    # ----------------------------------------
    C, g = new_pred(4)
    # basis_car = np.diag([0.01, 0, 0, 0])
    # center_car = np.array([0, 0, 0, 8])
    # init_car = StarSet(center_car, basis_car, C, g)

    # basis_ped = np.diag([0, 0, 0, 0])
    # center_ped = np.array([175, -55, 0, 3])
    # init_pedestrian = StarSet(center_ped, basis_ped, C, g)
    # ----------------------------------------

    basis_car = np.diag([0.01, 0.01, 0, 0])
    center_car = np.array([0, 0, 0, 8])
    init_car = StarSet(center_car, basis_car, C, g)

    basis_ped = np.diag([0, 0, 0, 0])
    center_ped = np.array([175, -55, 0, 3])
    init_pedestrian = StarSet(center_ped, basis_ped, C, g)

    # # Uncomment this block to use R1
    # -----------------------------------------
    # init_car = [[-5,-5,0,8],[5,5,0,8]]
    # init_pedestrian = [[175,-55,0,3],[175,-55,0,3]]
    # # -----------------------------------------

    # # Uncomment this block to use R2
    # init_car = [[-5,-5,0,7],[5,5,0,9]]
    # init_pedestrian = [[175,-55,0,3],[175,-55,0,3]]

    # init_car = [[-5,-5,0,7],[5,5,0,9]]
    # init_pedestrian = [[173,-55,0,3],[177,-55,0,3]]
    # init_car = from_rect(*init_car)
    # init_pedestrian = from_rect(*init_pedestrian)
    # # -----------------------------------------

    # # Uncomment this block to use R3
    # init_car = [[-5,-5,0,7],[5,5,0,9]]
    # init_pedestrian = [[173,-55,0,3],[176,-53,0,3]]
    # init_car = from_rect(*init_car)
    # init_pedestrian = from_rect(*init_pedestrian)
    # print(init_car.overapprox_rectangle(), init_pedestrian.overapprox_rectangle())
    # # -----------------------------------------

    # scenario.set_init_single(
    #     'car', init_car,(VehicleMode.Normal,)
    # )
    # scenario.set_init_single(
    #     'pedestrian', init_pedestrian, (PedestrianMode.Normal,)
    # )

    vehicle.set_initial(
        init_car, (VehicleMode.Normal,)
    )
    pedestrian.set_initial(
        init_pedestrian, (PedestrianMode.Normal,)
    )

    scenario.add_agent(vehicle) 
    scenario.add_agent(pedestrian)

    # trace = scenario.simulate_simple(50, 0.1)
    trace = scenario.verify(50, 0.2)
    # plot_reachtube_stars(trace, filter = 1)
    dist = []
    all_l_values = []
    all_t_values = []
    plt.rcParams.update({'font.size': 20})
    for node in trace.nodes:
        for i in range(0,len(node.trace['car'])):
            rect1 = (node.trace['car'][i][1].get_max_min(0)[0], node.trace['car'][i][1].get_max_min(1)[0], node.trace['car'][i][1].get_max_min(0)[1], node.trace['car'][i][1].get_max_min(1)[1])
            rect2 = (node.trace['pedestrian'][i][1].get_max_min(0)[0], node.trace['pedestrian'][i][1].get_max_min(1)[0], node.trace['pedestrian'][i][1].get_max_min(0)[1], node.trace['pedestrian'][i][1].get_max_min(1)[1])
            dist.append([node.trace['car'][i][0], get_extreme(rect1, rect2)])
    
        t_values = [t for t, _ in dist]
        l_values = [l_u[0] for _, l_u in dist]
        u_values = [l_u[1] for _, l_u in dist]
        plt.fill_between(t_values, l_values, u_values, alpha=0.3, color='b')
        # plt.plot(t_values, l_values, 'r--', label='Lower Bound (l)')
        # plt.plot(t_values, u_values, 'g--', label='Upper Bound (u)')
        dist = []
        all_l_values += l_values
        all_t_values += t_values
        # j+=1
        # if j==2:
        #     break 
    
    min_l = np.inf
    min_t = -1
    for i in range(len(all_t_values)):
        if all_t_values[i]>20:
            break
        if all_l_values[i]<min_l:
            min_l = all_l_values[i]
            min_t = all_t_values[i]
    # print(min_l, min_t)
    plt.title("Car Pedestrian Scenario Using StarredVerse")
    plt.ylabel("Relative Distance")
    plt.xlabel("Time (s)")
    plt.plot([], [], alpha=0.3, color='b', label='Range of Relative Distances')
    plt.axhline(y=2, color='r', linestyle='--', label='Unsafe Region Cut-off: $d$ = 2')
    plt.xlim([15, 20])
    plt.ylim([0,30])
    plt.scatter(min_t, min_l, color='red', marker='*')
    plt.annotate("$d_{min}$"+f"= {min_l:.2f}", xy=(min_t, min_l), xytext=(min_t - 1, min_l + 1),
             arrowprops=dict(arrowstyle="->", color='red'), color='red')
    plt.legend()
    plt.show()
    # plot_stars_time(trace)
    # avg_vel, unsafe_frac, unsafe_init = eval_velocity([trace]) # this probably breaks for star sets
    # fig = go.Figure()
    # fig = simulation_tree_3d(trace, fig,\
    #                          0,'time', 1,'x',2,'y')
    # fig.show()
