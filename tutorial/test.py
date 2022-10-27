import numpy as np
def car_dynamics(t, state, u):
    x, y, theta, v = state
    delta, a = u  
    x_dot = v*np.cos(theta+delta)
    y_dot = v*np.sin(theta+delta)
    theta_dot = v/1.75*np.tan(delta)
    v_dot = a 
    return [x_dot, y_dot, theta_dot, v_dot]

from tutorial_utils import car_action_handler
from typing import List 
import numpy as np 
from scipy.integrate import ode

def TC_simulate(mode: List[str], initialCondition, time_bound, time_step, track_map=None)->np.ndarray:
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound/time_step))
    t = [round(i*time_step,10) for i in range(0,number_points)]

    init = initialCondition
    trace = [[0]+init]
    for i in range(len(t)):
        steering, a = car_action_handler(mode, init, track_map)
        r = ode(car_dynamics)    
        r.set_initial_value(init).set_f_params([steering, a])      
        res:np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init) 

    return np.array(trace)

from verse.parser.parser import ControllerIR
from verse.agents import BaseAgent

class CarAgent(BaseAgent):
    def __init__(self, id, code = None, file_name = None):
        self.id = id 
        self.decision_logic = ControllerIR.parse(code, file_name)
        self.TC_simulate = TC_simulate

from enum import Enum, auto

class AgentMode(Enum):
    Normal = auto()
    Brake = auto()

class TrackMode(Enum):
    T0 = auto()

from verse.scenario import Scenario
from tutorial_map import M1
scenario = Scenario()
scenario.set_map(M1())


car1 = CarAgent('car1', file_name="./tutorial/dl_sec4.py")
car1.set_initial([[0,-0.5,0,2],[1,0.5,0,2]], (AgentMode.Normal, TrackMode.T0))
car2 = CarAgent('car2', file_name="./tutorial/dl_sec4.py")
car2.set_initial([[15,-0.5,0,1],[16,0.5,0,1]], (AgentMode.Normal, TrackMode.T0))
scenario.add_agent(car1)
scenario.add_agent(car2)

traces_simu = scenario.simulate(10, 0.01)
traces_veri = scenario.verify(10, 0.01)