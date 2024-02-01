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
from verse.map.lane_segment import StraightLane
from verse.map.lane import Lane
from math import pi
import random

def left():
    num = random.randint(1, 10)
    if num >=1 and num <= 9:
        return 0
    else:
        return 1

def right():
    num = random.randint(1, 10)
    if num >=1 and num <= 8:
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
                x = cont['ego.x']

                if(cont['ego.x']  < 7.5):
                    cont['ego.s'] = 0 #left() #0
                else:
                    cont['ego.s'] = 1 #right() #1

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

                if(cont['ego.x'][0] < 7.5):
                    state1 = 0#left() #0
                else:
                    state1 = 1#right() #1
                if(cont['ego.x'][1]  < 7.5):
                    state2 = 0#left() #0
                else:
                    state2 = 1 # right() #1
                if(state1 > state2):
                    state1 = state2
                cont['ego.s'] = [state1, state2]
                

        print(cont['ego.s'])

        if(cont['ego.s'] == [0,1]):
            print("hello")

        return cont, disc, len_dict

class M1(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = StraightLane("Seg0", [0, 0], [0, 100], 3)
        lane0 = Lane("T0", [segment0])
        self.add_lanes([lane0])
        self.h_dict = {("T0", "Left", "Right"): "T0", ("T0", "Right", "Left"): "T0",
                       ("T0", "Left", "Left"): "T0",("T0", "Right", "Right"): "T0"}

#modified T
def car_dynamics(t, state, u):
    x, y, theta, v, t1 = state
    delta, a, time_step = u
    x_dot = v * np.cos((theta*pi/180) + (delta*pi/180))
    y_dot = v * np.sin((theta*pi/180) + (delta*pi/180))
    theta_dot = delta#/time_step
    v_dot = a
    t_dot = 1
    return [x_dot, y_dot, theta_dot, v_dot, t_dot]

def car_action_handler(mode: List[str], state, lane_map) -> Tuple[float, float]:
    x, y, theta, v, t = state
    vehicle_mode = mode[0]
    vehicle_lane = mode[1]

    new_theta = theta
    if vehicle_mode == "Left":
        new_theta = 135
    elif vehicle_mode == "Right":
        new_theta = 45
    else:
        raise ValueError(f"Invalid mode: {vehicle_mode}")

    delta = new_theta - theta
    return delta, 0

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
        r.set_initial_value(init).set_f_params([steering, a, time_step])
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


scenario = Scenario(ScenarioConfig(parallel=False))
scenario.set_map(M1())
scenario.set_sensor(VechicleSensor())


car1 = CarAgent("car", file_name="sensedl.py")
car1.set_initial([[13, 0, 135, 3, 0], [14, 0, 135, 3, 0]], (AgentMode.Left, TrackMode.T0))
scenario.add_agent(car1)

traces_simu = scenario.simulate(30,0.01)
traces_veri = scenario.verify(30, 0.01)


fig = go.Figure()
fig = simulation_tree(traces_simu, None, fig, 1, 2, [0, 1], "lines", "trace")
fig.show()


fig = go.Figure()
fig = reachtube_tree(traces_veri, None, fig, 1, 2, [0, 1], "lines", "trace")
fig.show()





    

