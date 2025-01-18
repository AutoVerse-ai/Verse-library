from verse.map.lane_segment import StraightLane
from verse.map.lane import Lane

segment0 = StraightLane("seg0", [0, 0], [500, 0], 3)
lane0 = Lane("T0", [segment0])
segment1 = StraightLane("seg1", [0, 3], [500, 3], 3)
lane1 = Lane("T1", [segment1])

h_dict = {
    ("T0", "Normal", "SwitchLeft"): "M01",
    ("T1", "Normal", "SwitchRight"): "M10",
    ("M01", "SwitchLeft", "Normal"): "T1",
    ("M10", "SwitchRight", "Normal"): "T0",
}


def h(lane_idx, agent_mode_src, agent_mode_dest):
    return h_dict[(lane_idx, agent_mode_src, agent_mode_dest)]


def h_exist(lane_idx, agent_mode_src, agent_mode_dest):
    return (lane_idx, agent_mode_src, agent_mode_dest) in h_dict


from verse.map import LaneMap


class Map2Lanes(LaneMap):
    def __init__(self):
        super().__init__()
        self.add_lanes([lane0, lane1])
        self.h = h
        self.h_exist = h_exist


from enum import Enum, auto


class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    M01 = auto()
    M10 = auto()

#Sensor 1
class VechicleBaseSensor:
    def __init__(self):
        self.sensor = 1

    def sense(self, agent, state_dict, lane_map, simulate = True):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        if simulate:
            if agent.id == 'car1':
                cont['ego.x'] = state_dict['car1'][0][1]
                cont['ego.y'] = state_dict['car1'][0][2]
                cont['ego.theta'] = state_dict['car1'][0][3]
                cont['ego.v'] = state_dict['car1'][0][4]
                disc['ego.agent_mode'] = state_dict['car1'][1][0]
                disc['ego.track_mode'] = state_dict['car1'][1][1]

                cont['other.x'] = state_dict['car2'][0][1]
                cont['other.y'] = state_dict['car2'][0][2]
                disc['other.track_mode'] = state_dict['car2'][1][1]

            if agent.id == 'car2':
                cont['ego.x'] = state_dict['car2'][0][1]
                cont['ego.y'] = state_dict['car2'][0][2]
                cont['ego.theta'] = state_dict['car2'][0][3]
                cont['ego.v'] = state_dict['car2'][0][4]
                disc['ego.agent_mode'] = state_dict['car2'][1][0]
                disc['ego.track_mode'] = state_dict['car2'][1][1]

                cont['other.x'] = state_dict['car1'][0][1]
                cont['other.y'] = state_dict['car1'][0][2]
                disc['other.track_mode'] = state_dict['car1'][1][1]

        else:
            if agent.id == 'car1':
                cont['ego.x'] = [state_dict['car1'][0][0][1], state_dict['car1'][0][1][1]]
                cont['ego.y'] = [state_dict['car1'][0][0][2], state_dict['car1'][0][1][2]]
                cont['ego.theta'] = [state_dict['car1'][0][0][3], state_dict['car1'][0][1][3]]
                cont['ego.v'] = [state_dict['car1'][0][0][4], state_dict['car1'][0][1][4]]
                disc['ego.agent_mode'] = state_dict['car1'][1][0]
                disc['ego.track_mode'] = state_dict['car1'][1][1]

                #Sensor 1 Difference
                cont['other.x'] = [state_dict['car2'][0][0][1], state_dict['car2'][0][1][1]]
                cont['other.y'] = [state_dict['car2'][0][0][2], state_dict['car2'][0][1][2]]
                disc['other.track_mode'] = state_dict['car2'][1][1]

            if agent.id == 'car2':
                cont['ego.x'] = [state_dict['car2'][0][0][1], state_dict['car2'][0][1][1]]
                cont['ego.y'] = [state_dict['car2'][0][0][2], state_dict['car2'][0][1][2]]
                cont['ego.theta'] = [state_dict['car2'][0][0][3], state_dict['car2'][0][1][3]]
                cont['ego.v'] = [state_dict['car2'][0][0][4], state_dict['car2'][0][1][4]]
                disc['ego.agent_mode'] = state_dict['car2'][1][0]
                disc['ego.track_mode'] = state_dict['car2'][1][1]

                #Sensor 1 Difference
                cont['other.x'] = [state_dict['car1'][0][0][1], state_dict['car1'][0][1][1]] 
                cont['other.y'] = [state_dict['car1'][0][0][2], state_dict['car1'][0][1][2]]   
                disc['other.track_mode'] = state_dict['car1'][1][1]              
        return cont, disc, len_dict
    
#Sensor 2
class VechicleDistSensor:
    def __init__(self):
        self.sensor = 1

    def sense(self, agent, state_dict, lane_map, simulate = True):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        if simulate:
            if agent.id == 'car1':
                cont['ego.x'] = state_dict['car1'][0][1]
                cont['ego.y'] = state_dict['car1'][0][2]
                cont['ego.theta'] = state_dict['car1'][0][3]
                cont['ego.v'] = state_dict['car1'][0][4]
                disc['ego.agent_mode'] = state_dict['car1'][1][0]
                disc['ego.track_mode'] = state_dict['car1'][1][1]

                cont['other.dist'] = state_dict['car2'][0][1] - state_dict['car1'][0][1]
                disc['other.track_mode'] = state_dict['car2'][1][1]

            if agent.id == 'car2':
                cont['ego.x'] = state_dict['car2'][0][1]
                cont['ego.y'] = state_dict['car2'][0][2]
                cont['ego.theta'] = state_dict['car2'][0][3]
                cont['ego.v'] = state_dict['car2'][0][4]
                disc['ego.agent_mode'] = state_dict['car2'][1][0]
                disc['ego.track_mode'] = state_dict['car2'][1][1]

                cont['other.dist'] = state_dict['car1'][0][1] - state_dict['car2'][0][1]
                disc['other.track_mode'] = state_dict['car1'][1][1]

        else:
            if agent.id == 'car1':
                cont['ego.x'] = [state_dict['car1'][0][0][1], state_dict['car1'][0][1][1]]
                cont['ego.y'] = [state_dict['car1'][0][0][2], state_dict['car1'][0][1][2]]
                cont['ego.theta'] = [state_dict['car1'][0][0][3], state_dict['car1'][0][1][3]]
                cont['ego.v'] = [state_dict['car1'][0][0][4], state_dict['car1'][0][1][4]]
                disc['ego.agent_mode'] = state_dict['car1'][1][0]
                disc['ego.track_mode'] = state_dict['car1'][1][1]

                #Sensor 2 Difference
                disc['other.track_mode'] = state_dict['car2'][1][1]

                other_x = [state_dict['car2'][0][0][1], state_dict['car2'][0][1][1]]
                other_y = [state_dict['car2'][0][0][2], state_dict['car2'][0][1][2]]

                ego_long = []
                other_long = []
                for x in cont['ego.x']:
                    for y in cont['ego.y']:
                        ego_long.append(lane_map.get_longitudinal_position(disc['ego.track_mode'], [x,y]))

                for x in other_x:
                    for y in other_y:
                        other_long.append(lane_map.get_longitudinal_position(disc['other.track_mode'], [x,y]))

                cont['other.dist'] = [min(other_long) - max(ego_long), max(other_long) - min(ego_long)]

                

            if agent.id == 'car2':
                cont['ego.x'] = [state_dict['car2'][0][0][1], state_dict['car2'][0][1][1]]
                cont['ego.y'] = [state_dict['car2'][0][0][2], state_dict['car2'][0][1][2]]
                cont['ego.theta'] = [state_dict['car2'][0][0][3], state_dict['car2'][0][1][3]]
                cont['ego.v'] = [state_dict['car2'][0][0][4], state_dict['car2'][0][1][4]]
                disc['ego.agent_mode'] = state_dict['car2'][1][0]
                disc['ego.track_mode'] = state_dict['car2'][1][1]

                #Sensor 2 Difference
                disc['other.track_mode'] = state_dict['car1'][1][1]

                other_x = [state_dict['car1'][0][0][1], state_dict['car1'][0][1][1]]
                other_y = [state_dict['car1'][0][0][2], state_dict['car1'][0][1][2]]

                ego_long = []
                other_long = []
                for x in cont['ego.x']:
                    for y in cont['ego.y']:
                        ego_long.append(lane_map.get_longitudinal_position(disc['ego.track_mode'], [x,y]))

                for x in other_x:
                    for y in other_y:
                        other_long.append(lane_map.get_longitudinal_position(disc['other.track_mode'], [x,y]))

                cont['other.dist'] = [min(other_long) - max(ego_long), max(other_long) - min(ego_long)]              
        return cont, disc, len_dict

#Sensor 3
class VechicleNoisySensor:
    def __init__(self, noise):
        self.sensor = 1
        self.noise = noise

    def sense(self, agent, state_dict, lane_map, simulate = True):
        len_dict = {}
        cont = {}
        disc = {}
        len_dict = {"others": len(state_dict) - 1}
        if simulate:
            if agent.id == 'car1':
                cont['ego.x'] = state_dict['car1'][0][1]
                cont['ego.y'] = state_dict['car1'][0][2]
                cont['ego.theta'] = state_dict['car1'][0][3]
                cont['ego.v'] = state_dict['car1'][0][4]
                disc['ego.agent_mode'] = state_dict['car1'][1][0]
                disc['ego.track_mode'] = state_dict['car1'][1][1]

                cont['other.dist'] = (state_dict['car2'][0][1] - state_dict['car1'][0][1]) + self.noise
                disc['other.track_mode'] = state_dict['car2'][1][1]

            if agent.id == 'car2':
                cont['ego.x'] = state_dict['car2'][0][1]
                cont['ego.y'] = state_dict['car2'][0][2]
                cont['ego.theta'] = state_dict['car2'][0][3]
                cont['ego.v'] = state_dict['car2'][0][4]
                disc['ego.agent_mode'] = state_dict['car2'][1][0]
                disc['ego.track_mode'] = state_dict['car2'][1][1]

                cont['other.dist'] = (state_dict['car1'][0][1] - state_dict['car2'][0][1]) + self.noise
                disc['other.track_mode'] = state_dict['car1'][1][1]

        else:
            if agent.id == 'car1':
                cont['ego.x'] = [state_dict['car1'][0][0][1], state_dict['car1'][0][1][1]]
                cont['ego.y'] = [state_dict['car1'][0][0][2], state_dict['car1'][0][1][2]]
                cont['ego.theta'] = [state_dict['car1'][0][0][3], state_dict['car1'][0][1][3]]
                cont['ego.v'] = [state_dict['car1'][0][0][4], state_dict['car1'][0][1][4]]
                disc['ego.agent_mode'] = state_dict['car1'][1][0]
                disc['ego.track_mode'] = state_dict['car1'][1][1]

                #Sensor 3 Difference
                disc['other.track_mode'] = state_dict['car2'][1][1]

                other_x = [state_dict['car2'][0][0][1], state_dict['car2'][0][1][1]]
                other_y = [state_dict['car2'][0][0][2], state_dict['car2'][0][1][2]]

                ego_long = []
                other_long = []
                for x in cont['ego.x']:
                    for y in cont['ego.y']:
                        ego_long.append(lane_map.get_longitudinal_position(disc['ego.track_mode'], [x,y]))

                for x in other_x:
                    for y in other_y:
                        other_long.append(lane_map.get_longitudinal_position(disc['other.track_mode'], [x,y]))

                cont['other.dist'] = [min(other_long) - max(ego_long), max(other_long) - min(ego_long)]
                cont['other.dist'] = [cont['other.dist'][0] - self.noise,  cont['other.dist'][1] + self.noise]

            if agent.id == 'car2':
                cont['ego.x'] = [state_dict['car2'][0][0][1], state_dict['car2'][0][1][1]]
                cont['ego.y'] = [state_dict['car2'][0][0][2], state_dict['car2'][0][1][2]]
                cont['ego.theta'] = [state_dict['car2'][0][0][3], state_dict['car2'][0][1][3]]
                cont['ego.v'] = [state_dict['car2'][0][0][4], state_dict['car2'][0][1][4]]
                disc['ego.agent_mode'] = state_dict['car2'][1][0]
                disc['ego.track_mode'] = state_dict['car2'][1][1]

                #Sensor 3 Difference
                disc['other.track_mode'] = state_dict['car1'][1][1]

                other_x = [state_dict['car1'][0][0][1], state_dict['car1'][0][1][1]]
                other_y = [state_dict['car1'][0][0][2], state_dict['car1'][0][1][2]]

                ego_long = []
                other_long = []
                for x in cont['ego.x']:
                    for y in cont['ego.y']:
                        ego_long.append(lane_map.get_longitudinal_position(disc['ego.track_mode'], [x,y]))

                for x in other_x:
                    for y in other_y:
                        other_long.append(lane_map.get_longitudinal_position(disc['other.track_mode'], [x,y]))

                cont['other.dist'] = [min(other_long) - max(ego_long), max(other_long) - min(ego_long)]
                cont['other.dist'] = [cont['other.dist'][0] - self.noise,  cont['other.dist'][1] + self.noise]            
        return cont, disc, len_dict


from verse.scenario import Scenario, ScenarioConfig

scenario = Scenario(ScenarioConfig(parallel=False))
scenario.set_map(Map2Lanes())

from tutorial_agent import CarAgent

car1 = CarAgent("car1", file_name="tutorial/dl_sec6.py")
car1.set_initial([[0, -0.5, 0, 2], [0.5, 0.5, 0, 2]], (AgentMode.Normal, TrackMode.T0))
car2 = CarAgent("car2", file_name="tutorial/dl_sec6.py")
car2.set_initial([[20, -0.5, 0, 1], [20.5, 0.5, 0, 1]], (AgentMode.Normal, TrackMode.T0))
scenario.add_agent(car1)
scenario.add_agent(car2)

#scenario.set_sensor(VechicleBaseSensor())
scenario.set_sensor(VechicleDistSensor())
#scenario.set_sensor(VechicleNoisySensor(1))

traces_veri = scenario.verify(20, 0.01)

import plotly.graph_objects as go
from verse.plotter.plotter2D import *

fig = go.Figure()
fig = reachtube_tree(traces_veri, Map2Lanes(), fig, 1, 2, [1, 2], "lines", "trace")
#fig.show()
fig.write_html('figure.html', auto_open=True)


