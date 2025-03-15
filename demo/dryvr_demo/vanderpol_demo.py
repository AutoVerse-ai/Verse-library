from origin_agent import vanderpol_agent
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.plotter.plotter2D import *


from verse.stars.starset import *
import verse.stars.starset
import plotly.graph_objects as go
from enum import Enum, auto

from verse.sensor.base_sensor_stars import *
import time
from verse.utils.star_diams import *
from scipy.spatial import HalfspaceIntersection
import hopsy

class AgentMode(Enum):
    Default = auto()

def plot_stars(stars: List[StarSet], dim1: int = 0, dim2: int = 1):
    for star in stars:
        x, y = np.array(star.get_verts(dim1, dim2))
        plt.plot(x, y, lw = 1)
        centerx, centery = star.get_center_pt(0, 1)
        plt.plot(centerx, centery, 'o')
    plt.show()


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/vanderpol_controller.py"
    scenario = Scenario(ScenarioConfig(parallel=False))
    scenario.config.model_path = 'vdp_svd_bench'

    scenario.config.model_hparams = {
        "big_initial_set": (np.array([0,-0.5,0,0,0,0]), np.array([15,0.5,0,0,0,0])), # irrelevant for now
        "initial_set_size": 1,
        "lamb": 7,
        "num_epochs": 30,
        "gamma":0.99,
        "lr":1e-4,
        "sublin_loss":True,
        # "num_samples": 100,
        # "Ns": 1
    }

    car = vanderpol_agent("car1", file_name=input_code_name)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    basis = np.diag([0.15, 0.05])
    center = np.array([1.40,2.3])
    # basis = np.diag([0.005, 0.005])
    # center = np.array([1.255, 2.255])
    C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
    g = np.array([1,1,1,1])

    # initial_trans = StarSet(center-np.array([-1, 1]), basis@np.array([[-1, 1], [1, 0]]), np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]])), g)
    # plot_stars([initial, initial_trans])
    ### how do I instantiate a scenario with a starset instead of a hyperrectangle?
    car.set_initial(
            # [[1.25, 2.25], [1.25, 2.25]],
            # [[1.55, 2.35], [1.55, 2.35]]
            StarSet(center, basis, C, g)
        ,
            tuple([AgentMode.Default])
            # tuple([AgentMode.Default]),
        ,
    )

    scenario.add_agent(car)
    # scenario.config.overwrite = True
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.set_sensor(BaseStarSensor())
    # scenario.config.overwrite = True

    # initial = StarSet(center, basis, C, g)
    # initial = StarSet(center, np.zeros((2,2)), C, g)
    initial = StarSet(center, np.array([[1,1], [2,2]]), C, g)
    # samples = initial.sample_h(num_samples=100)
    # plt.scatter(samples[:,0], samples[:,1])
    # print(samples)
    # plot_stars([initial])
    # exit()
    start = time.time()
    traces = scenario.verify(7, 0.1)
    end = time.time()
    print(f'Run time: {end-start}')
    diams = time_step_diameter(traces, 7, 0.1)
    print(f'Initial diameter: {diams[0]}\n Final: {diams[-1]}\n Average: {sum(diams)/len(diams)}')
    # print(sum(diams), '\n', len(diams))
    # exit()
    car1 = traces.nodes[0].trace['car1']
    car1 = [star[1] for star in car1]
    car0 = car1[0]
    # v = np.array(car0.get_verts()).T
    # print(car0.get_verts_dim())

    # for star in car1:
    #     print(star.center, star.basis, star.C, star.g, '\n --------')
    # plot_stars(car1, 0, 1)
    # S = sample_star(StarSet(center, basis, C, g), N=30)
    # St = []
    # for p in S: 
    #     St.append(car.TC_simulate((AgentMode.Default), p, 7, 0.1).tolist())
    # St = np.array(St) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]
    # for t in range(len(St[0])):
    #     plt.scatter(St[:,t,1], St[:,t,2], color='black') 
    plot_reachtube_stars(traces, filter=1)

    # plot_stars_time(traces, 0, sim=vanderpol_agent.TC_simulate, scenario_agent=car)
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 1, [0, 1], "lines", "trace")
    # fig.show()
