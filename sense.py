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

                x = cont['ego.x']
                y = cont['ego.y']
                position = np.array([x,y])
                lateral = lane_map.get_lateral_distance("T0", position)

                if(lateral < -1):
                    cont['ego.s'] = 0
                elif(lateral < 0):
                    cont['ego.s'] = slightly_left()
                elif(lateral < 1):
                    cont['ego.s'] = slightly_right()
                else:
                    cont['ego.s'] = 1

        else:
            if agent.id == 'car':
                cont['ego.x'] = [state_dict['car'][0][0][1], state_dict['car'][0][1][1]]
                cont['ego.y'] = [state_dict['car'][0][0][2], state_dict['car'][0][1][2]]
                cont['ego.theta'] = [state_dict['car'][0][0][3], state_dict['car'][0][1][3]]
                cont['ego.v'] = [state_dict['car'][0][0][4], state_dict['car'][0][1][4]]
                cont['ego.t'] = [state_dict['car'][0][0][5], state_dict['car'][0][1][5]]
                disc['ego.agent_mode'] = state_dict['car'][1][0]
                
                lateral = []
                for x in cont['ego.x']:
                    for y in cont['ego.y']:
                        position = np.array([x,y])
                        print(position)
                        lateral.append(lane_map.get_lateral_distance("T0", position))

                min_lateral = min(lateral)

                if(min_lateral < -1):
                    cont['ego.s'] = [0,0]
                elif(min_lateral < 0):
                    cont['ego.s'] = [0,1]
                elif(min_lateral < 1):
                    cont['ego.s'] = [0,1]
                else:
                    cont['ego.s'] = [1,1]

                
        return cont, disc, len_dict

class M1(LaneMap):
    def __init__(self):
        super().__init__()
        segment0 = CircularLane("Seg0", [14, 14], 12, np.pi * 3 / 2, np.pi, True, 4)
        segment1 = StraightLane("Seg1", [2, 14], [2, 24], 4)
        segment2 = CircularLane("Seg2", [14, 24], 12, np.pi, 0, True, 4)
        segment3 = StraightLane("Seg3", [26, 24], [26, 14], 4)
        lane0 = Lane("T0", [segment0, segment1, segment2])# segment3])
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


scenario = Scenario(ScenarioConfig(parallel=False))
scenario.set_map(M1())
scenario.set_sensor(VechicleSensor())


car1 = CarAgent("car", file_name="sensedl.py")
car1.set_initial([[14, 1, np.radians(180), 0.75, 0], [14, 2, np.radians(180), 0.75, 0]], (AgentMode.Right, TrackMode.T0))
scenario.add_agent(car1)

#traces_simu = scenario.simulate(60,0.01)
traces_veri = scenario.verify(10, 0.01)

traces_veri.dump("out.json")

# fig = go.Figure()
# fig = simulation_tree(traces_simu, None, fig, 1, 2, [0, 1], "lines", "trace")
# fig.show()



fig = go.Figure()
fig = reachtube_tree(traces_veri, None, fig, 1, 2, [0, 1], "lines", "trace")
fig.show()




    

