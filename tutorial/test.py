from tutorial_map import M1

map1 = M1()

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
import copy\n\
def decisionLogic(ego:State, track_map):\n\
    output = copy.deepcopy(ego)\n\
    if ego.agent_mode == AgentMode.Normal:\n\
        if ego.x > 10:\n\
            output.agent_mode = 'Brake'\n\
    return output\n\
"

from tutorial_agent import Agent1

car = Agent1('car', code=dl)

from verse.scenario import Scenario

scenario = Scenario()

scenario.add_agent(car)

scenario.set_map(map1)

scenario.set_init([[[0,-0.5,0,1],[1,0.5,0,1]]], [(AgentMode.Normal, TrackMode.T0)])

# traces_sim = scenario.simulate(15, 0.1)
traces_veri = scenario.verify(15, 0.1)

import plotly.graph_objects as go
from verse.plotter.plotter2D import *

fig = go.Figure()
fig = reachtube_tree(traces_veri, None, fig, 0, 1, [0, 1], 'lines', 'trace')
fig.show()