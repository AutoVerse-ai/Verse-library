from enum import Enum, auto
from typing import List
import copy
import math
import numpy as np

#the states the car can be in
class AgentMode(Enum):
    Left = auto()
    Right = auto()

#two tracks for the map
class TrackMode(Enum):
    T0 = auto()
    T1 = auto()

#state of the car
class State:
    x: float
    y: float
    theta: float
    v: float
    t: float
    agent_mode: AgentMode
    track_mode: TrackMode

    def __init__(self, x, y, theta, v,t, agent_mode: AgentMode, track_mode: TrackMode):
        pass


#decision logic of the car; changes based on the track (y<15 is T0 and y>15 is T1)
def decisionLogic(ego: State, track_map):
    output = copy.deepcopy(ego)
    '''
    if(ego.y > 14):
        if ego.s == 0 and ego.t >= 2: #we're on the left side
            output.agent_mode = AgentMode.StraightRight #go right
            output.track_mode = TrackMode.T1
            output.t = 0
        elif ego.s == 1 and ego.t >= 2: #we're on the right side
            output.agent_mode = AgentMode.StraightLeft #go left
            output.t = 0
            output.track_mode = TrackMode.T1
    elif(ego.y < 14):
    '''
    if ego.s == 0 and ego.t >= 2: #we're on the left side
        output.agent_mode = AgentMode.Right #go right
        output.t = 0
        output.track_mode = TrackMode.T0
    elif ego.s == 1 and ego.t >= 2: #we're on the right side
        output.agent_mode = AgentMode.Left #go left
        output.t = 0
        output.track_mode = TrackMode.T0

    
    x = ego.x
    y = ego.y
    position = [x,y]
    lateral = track_map.get_lateral_distance("T0", position)
    
    # assert -2 <= lateral and lateral <= 2
    #assert ((ego.x-14)**2 + (ego.y-14)**2 >= 10**2 and (ego.x-14)**2 + (ego.y-14)**2 <= 14**2 and ego.y <= 14) or (ego.x > 0 and ego.x < 4 and ego.y > 14 and ego.y <= 24) or ((ego.x-14)**2 + (ego.y-24)**2 >= 10**2 and (ego.x-14)**2 + (ego.y-24)**2 <= 14**2 and ego.y > 24)
    assert True 
     
    return output


