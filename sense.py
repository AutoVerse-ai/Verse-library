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
            if agent.id == 'car':
                
                cont['ego.x'] = state_dict['car'][0][1]
                cont['ego.y'] = state_dict['car'][0][2]
                cont['ego.theta'] = state_dict['car'][0][3]
                cont['ego.v'] = state_dict['car'][0][4]
                cont['ego.t'] = state_dict['car'][0][5]
                disc['ego.agent_mode'] = state_dict['car'][1][0]

                x = state_dict['car'][0][1]#cont['ego.x']
                y = state_dict['car'][0][2]#cont['ego.y']
                agent_mode = state_dict['car'][1][0]

                #print(agent_mode)

                if(agent_mode == "Left" or agent_mode == "Right"):
                    if((x-14)**2 + (y-14)**2 > 13**2):
                        cont['ego.s'] = very_left()
                    elif((x-14)**2 + (y-14)**2 > 12**2):
                        cont['ego.s'] = slightly_left()
                    elif((x-14)**2 + (y-14)**2 > 11**2):
                        cont['ego.s'] = slightly_right()
                    else:
                        cont['ego.s'] = very_right()
                else:
                    if(x < 1):
                        cont['ego.s'] = very_left()
                    elif(x < 2):
                        cont['ego.s'] = slightly_left()
                    elif(x < 3):
                        cont['ego.s'] = slightly_right()
                    else:
                        cont['ego.s'] = very_right()

        else:
            if agent.id == 'car':
                cont['ego.x'] = [state_dict['car'][0][0][1], state_dict['car'][0][1][1]]
                cont['ego.y'] = [state_dict['car'][0][0][2], state_dict['car'][0][1][2]]
                cont['ego.theta'] = [state_dict['car'][0][0][3], state_dict['car'][0][1][3]]
                cont['ego.v'] = [state_dict['car'][0][0][4], state_dict['car'][0][1][4]]
                cont['ego.t'] = [state_dict['car'][0][0][5], state_dict['car'][0][1][5]]
                disc['ego.agent_mode'] = state_dict['car'][1][0]
                
                state1 = -1
                state2 = -1

                x = cont['ego.x'][0]
                y = cont['ego.y'][0]
                agent_mode = state_dict['car'][1][0]


                if(agent_mode == "Left" or agent_mode == "Right"):
                    if((x-14)**2 + (y-14)**2 > 13**2):
                        state1 = very_left()
                    elif((x-14)**2 + (y-14)**2 > 12**2):
                        state1 = slightly_left()
                    elif((x-14)**2 + (y-14)**2 > 11**2):
                        state1 = slightly_right()
                    else:
                        state1 = very_right()
                else:
                    if(x < 1):
                        state1 = very_left()
                    elif(x < 2):
                        state1 = slightly_left()
                    elif(x < 3):
                        state1 = slightly_right()
                    else:
                        state1 = very_right()

                x = cont['ego.x'][1]
                y = cont['ego.y'][1]

                if(agent_mode == "Left" or agent_mode == "Right"):
                    if((x-14)**2 + (y-14)**2 > 13**2):
                        state2 = very_left()
                    elif((x-14)**2 + (y-14)**2 > 12**2):
                        state2 = slightly_left()
                    elif((x-14)**2 + (y-14)**2 > 11**2):
                        state2 = slightly_right()
                    else:
                        state2 = very_right()
                else:
                    if(x < 1):
                        state2 = very_left()
                    elif(x < 2):
                        state2 = slightly_left()
                    elif(x < 3):
                        state2 = slightly_right()
                    else:
                        state2 = very_right()

                if(state1 > state2):
                    state1 = state2
                cont['ego.s'] = [state1, state2]
                
        '''
        print(cont['ego.s'])

        if(cont['ego.s'] == [0,1]):
            print("hello")
        '''
        return cont, disc, len_dict

class M1(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = CircularLane("Seg0", [14, 14], 12, np.pi * 3 / 2, np.pi * (-7/2), True, 4)
        segment1 = StraightLane("Seg1", [0, 10], [0, 100], 4)
        lane0 = Lane("T0", [segment0])
        lane1 = Lane("T1", [segment1])
        self.add_lanes([lane0, lane1])
        self.h_dict = {("T0", "Left", "Right"): "T0", ("T0", "Right", "Left"): "T0",
                       ("T0", "Left", "Left"): "T0",("T0", "Right", "Right"): "T0",

                       ("T0", "Left", "StraightLeft"): "T1", ("T0", "Right", "StraightRight"): "T1",
                       ("T0", "Left", "StraightRight"): "T1",("T0", "Right", "StraightLeft"): "T1",

                       ("T1", "StraightLeft", "StraightRight"): "T1", ("T1", "StraightLeft", "StraightLeft"): "T1",
                       ("T1", "StraightRight", "StraightRight"): "T1", ("T1", "StraightRight", "StraightLeft"): "T1"}


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

    '''
    center = np.array([14,14])
    v = pos - center
    if(np.sum(v) == 0):
        track_pos = pos
    else:
        track_pos = center + (v/np.linalg.norm(v))*12
    '''

    new_theta = 0
    if vehicle_mode == "Left" or vehicle_mode == "StraightLeft":
        new_theta = 45
    elif vehicle_mode == "Right" or vehicle_mode == "StraightRight":
        new_theta = -45
    else:
        raise ValueError(f"Invalid mode: {vehicle_mode}")

    #print(vehicle_lane)

    new_theta = np.radians(new_theta)

    if(vehicle_lane == "T0"):
        heading = lane_map.get_lane_heading(vehicle_lane, pos) + np.arctan2(0.45 * new_theta, v)
        psi = wrap_to_pi(heading - theta)
    else: 
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
car1.set_initial([[14, 1, np.radians(180), 0.75, 0], [14, 1, np.radians(180), 0.75, 0]], (AgentMode.Right, TrackMode.T0))
scenario.add_agent(car1)

traces_simu = scenario.simulate(40,0.01)
#traces_veri = scenario.verify(40, 0.01)


fig = go.Figure()
fig = simulation_tree(traces_simu, None, fig, 1, 2, [0, 1], "lines", "trace")
fig.show()


fig = go.Figure()
fig = reachtube_tree(traces_veri, None, fig, 1, 2, [0, 1], "lines", "trace")
fig.show()




    

