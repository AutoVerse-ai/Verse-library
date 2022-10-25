from verse.scenario import Scenario
from tutorial_map import M1
scenario = Scenario()
scenario.config.init_seg_length = 5
scenario.set_map(M1())

from enum import Enum, auto

class AgentMode(Enum):
    Normal = auto()
    Brake = auto()
    
class TrackMode(Enum):
    T0 = auto()
    
class State:
    x:float
    y:float
    theta:float
    v:float
    agent_mode:AgentMode 
    track_mode:TrackMode 

    def __init__(self, x, y, theta, v, agent_mode: AgentMode, track_mode: TrackMode):
        pass

from typing import List
import copy
def decisionLogic(ego:State, others: List[State], track_map):
    output = copy.deepcopy(ego)
    if ego.agent_mode == AgentMode.Normal:
        if any( other.x-ego.x< 8 and other.x-ego.x>0 for other in others):
            output.agent_mode = AgentMode.Brake
    
    ### Adding safety assertions
    assert not any(other.x-ego.x<1.0 and other.x-ego.x>-1.0 for other in others), 'Seperation'
    ##########
    
    return output

dl = "from enum import Enum, auto\n\
\n\
class AgentMode(Enum):\n\
    Normal = auto()\n\
    Brake = auto()\n\
\n\
class TrackMode(Enum):\n\
    T0 = auto()\n\
\n\
class State:\n\
    x:float\n\
    y:float\n\
    theta:float\n\
    v:float\n\
    agent_mode:AgentMode\n\
    track_mode:TrackMode \n\
\n\
    def __init__(self, x, y, theta, v, agent_mode: AgentMode, track_mode: TrackMode):\n\
        pass\n\
\n\
from typing import List\n\
import copy\n\
def decisionLogic(ego:State, others: List[State], track_map):\n\
    output = copy.deepcopy(ego)\n\
    if ego.agent_mode == AgentMode.Normal:\n\
        if any( other.x-ego.x< 8 and other.x-ego.x>0 for other in others):\n\
            output.agent_mode = AgentMode.Brake\n\
\n\
    ### Adding safety assertions\n\
    assert not any(other.x-ego.x<1.0 and other.x-ego.x>-1.0 for other in others), 'Seperation'\n\
    ##########\n\
\n\
    return output\n\
"

from tutorial_agent import Agent1
car1 = Agent1('car1', code=dl)
car1.set_initial([[0,-0.5,0,2],[1,0.5,0,2]], (AgentMode.Normal, TrackMode.T0))
car2 = Agent1('car2', code=dl)
car2.set_initial([[15,-0.5,0,1],[16,0.5,0,1]], (AgentMode.Normal, TrackMode.T0))
scenario.add_agent(car1)
scenario.add_agent(car2)
from tutorial_sensor import DefaultSensor
scenario.set_sensor(DefaultSensor())

scenario.set_init_single('car1',[[0,-0.5,0,5],[1,0.5,0,5]], (AgentMode.Normal, TrackMode.T0))

traces_simu = scenario.simulate(10, 0.01)
traces_veri = scenario.verify(10, 0.01)

import plotly.graph_objects as go
from verse.plotter.plotter2D import *

fig = go.Figure()
fig = simulation_tree(traces_simu, None, fig, 0, 1, [0, 1], 'lines', 'trace')
fig.show()

fig = go.Figure()
fig = reachtube_tree(traces_veri, None, fig, 0, 1, [0, 1], 'lines', 'trace')
fig.show()