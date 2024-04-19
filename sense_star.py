import numpy as np
from typing import List
from scipy.integrate import ode
from verse.scenario import Scenario, ScenarioConfig
from verse.parser.parser import ControllerIR
from verse.agents import BaseAgent
from verse.map import *
from sensedl import *
import plotly.graph_objects as go
from verse.plotter.plotter2D import *
from verse.map.lane_map import LaneMap
from verse.map.lane_segment import StraightLane, CircularLane
from verse.map.lane import Lane
from math import pi
import random
from verse.analysis.utils import wrap_to_pi 
from verse.analysis import ReachabilityMethod
import numpy as np
from verse.map.lane_segment import StraightLane
from verse.map.lane import Lane
from verse.starsproto.starset import StarSet
import polytope as pc


#probabilistic sensors
def very_left():
    num = random.randint(1,100)
    if num >=1 and num <= 98:
        return 0
    else:
        return 1
def slightly_left():
    num = random.randint(1, 100)
    if num >=1 and num <= 90:
        return 0
    else:
        return 1
def slightly_right():
    num = random.randint(1, 100)
    if num >=1 and num <= 80:
        return 1
    else:
        return 0  
def very_right():
    num = random.randint(1, 100)
    if num >=1 and num <= 95:
        return 1
    else:
        return 0


class VechicleSensor:
    def __init__(self):
        self.sensor = 1

    # The baseline sensor is omniscient. Each agent can get the state of all other agents
    def sense(self, agent: BaseAgent, state_dict, lane_map):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        tmp = np.array(list(state_dict.values())[0][0])
        if tmp.ndim < 2:
            num = 1
            if agent.id == 'car':
                car = state_dict['car']
                star = car[0][1]
                rect = star.overapprox_rectangle()

                cont['ego.x'] = star
                cont['ego.y'] = star
                cont['ego.theta'] = star
                cont['ego.v'] = star
                cont['ego.t'] = star
                disc['ego.agent_mode'] = state_dict['car'][1][0]


                xlist = [rect[0][0], rect[1][0]]
                ylist = [rect[0][1], rect[1][1]]

                true_lateral = []
                for x in xlist:
                    for y in ylist:
                        position = np.array([x,y])
                        #print(position)
                        true_lateral.append(lane_map.get_lateral_distance("T0", position))

                pol = pc.box2poly([[min(true_lateral), min(true_lateral)], [min(true_lateral), min(true_lateral)]]) 
                cont['ego.s'] = StarSet.from_polytope(pol)


        else:
            if agent.id == 'car':

                car = state_dict['car']

                star1 = car[0][0][1]
                star2 = car[0][1][1]
                
                rect1 = star1.overapprox_rectangle()
                rect2 = star2.overapprox_rectangle()

                cont['ego.x'] = [star1, star2]
                cont['ego.y'] = [star1, star2]
                cont['ego.theta'] = [star1, star2]
                cont['ego.v'] = [star1, star2]
                cont['ego.t'] = [star1, star2]

                disc['ego.agent_mode'] = state_dict['car'][1][0]

                xlist = [rect2[0][0], rect2[1][0]]
                ylist = [rect2[0][1], rect2[1][1]]

                # print(cont)
                # print(disc)

                # print(xlist)
                # print(ylist)


                true_lateral = []
                for x in xlist:
                    for y in ylist:
                        position = np.array([x,y])
                        #print(position)
                        true_lateral.append(lane_map.get_lateral_distance("T0", position))

                blowup = [[min(true_lateral) - 0.005, min(true_lateral) - 0.005],[ max(true_lateral) + 0.005,  max(true_lateral) + 0.005]]
                #perception contract: blow up lateral array and choose minimum

                pol = pc.box2poly(blowup)
                new_star = StarSet.from_polytope(pol)
                cont['ego.s'] = [new_star, new_star]


                
        return cont, disc, len_dict

class M1(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = CircularLane("Seg0", [14, 14], 12, np.pi * 3 / 2, np.pi, True, 4)
        segment1 = StraightLane("Seg1", [2, 14], [2, 24], 4)
        lane0 = Lane("T0", [segment0, segment1])
        self.add_lanes([lane0])
        self.h_dict = {("T0", "Left", "Right"): "T0", ("T0", "Right", "Left"): "T0",
                       ("T0", "Left", "Left"): "T0",("T0", "Right", "Right"): "T0"}

def car_dynamics(t, state, u):
    x, y, theta, v, t = state
    delta, a = u
    x_dot = v * np.cos(theta + delta)
    y_dot = v * np.sin(theta + delta)
    theta_dot = v / 1.75 * np.sin(delta)
    v_dot = a
    t_dot = 1
    return [x_dot, y_dot, theta_dot, v_dot, t_dot]

def car_action_handler(mode: List[str], state, lane_map) -> Tuple[float, float]:
    x, y, theta, v, t = state
    vehicle_mode = mode[0]
    vehicle_lane = mode[1]
    heading  = 0
    d = 0
    pos = np.array([x,y])

    new_theta = 0
    if vehicle_mode == "Left":
        new_theta = 45
    elif vehicle_mode == "Right":
        new_theta = -45
    else:
        raise ValueError(f"Invalid mode: {vehicle_mode}")


    new_theta = np.radians(new_theta)
    heading = lane_map.get_lane_heading(vehicle_lane, pos) + np.arctan2(0.45 * new_theta, v)
    psi = wrap_to_pi(heading - theta)
    steering = psi
    steering = np.clip(steering, -0.4, 0.4)
    return steering, 0

def TC_simulate(
    mode: List[str], init, time_bound, time_step, lane_map: LaneMap = None
):
    time_bound = float(time_bound)
    num_points = int(np.ceil(time_bound / time_step))
    trace = np.zeros((num_points + 1, 1 + len(init)))
    trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
    trace[0, 1:] = init
    for i in range(num_points):
        steering, a = car_action_handler(mode, init, lane_map)
        r = ode(car_dynamics)
        r.set_initial_value(init).set_f_params([steering, a])
        res: np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten()
        if init[3] < 0:
            init[3] = 0
        trace[i + 1, 0] = time_step * (i + 1)
        trace[i + 1, 1:] = init
    return trace

class CarAgent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        self.id = id
        self.decision_logic = ControllerIR.parse(code, file_name)
        self.TC_simulate = TC_simulate


scenario = Scenario()#ScenarioConfig(parallel=False))
scenario.set_map(M1())
scenario.set_sensor(VechicleSensor())


car1 = CarAgent("car", file_name="sensedl.py")
initial_set_polytope = pc.box2poly([[14, 14.1], [1.75, 2], [np.radians(180),np.radians(180)], [0.75,0.75], [0,0]])
# car1.set_initial([[14, 1.75, np.radians(180), 0.75, 0], [14, 2, np.radians(180), 0.75, 0]], (AgentMode.Right, TrackMode.T0))
car1.set_initial(StarSet.from_polytope(initial_set_polytope), (AgentMode.Right, TrackMode.T0))

scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
scenario.add_agent(car1)

# traces_simu = scenario.simulate(35,0.01)

# fig = go.Figure()
# fig = simulation_tree(traces_simu, None, fig, 1, 2, [0, 1], "lines", "trace")
# fig.show()

traces_veri = scenario.verify(10, 0.01)

# fig = go.Figure()
# fig = reachtube_tree(traces_veri, None, fig, 1, 2, [0, 1], "lines", "trace")
# fig.show()

#nodes = traces_veri._get_all_nodes(traces_veri.root)
#print(len(nodes))
# for n in nodes:
#     print("new node")
#     for l in n.trace['car']:
#         print(l[1:3])

#traces_veri.visualize()
#print(nodes)
#print("height")
#print(traces_veri.height)

#traces_veri.dump("out.json")

import plotly.graph_objects as go
from verse.plotter.plotterStar import *

plot_reachtube_stars(traces_veri, M1(), 0 , 1, 10)