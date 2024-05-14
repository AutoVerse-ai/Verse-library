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


from verse.scenario import Scenario

scenario = Scenario()
scenario.set_map(Map2Lanes())

from tutorial_agent import CarAgent

car1 = CarAgent("car1", file_name="./dl_sec5.py")
car1.set_initial([[0, -0.5, 0, 2], [0.5, 0.5, 0, 2]], (AgentMode.Normal, TrackMode.T0))
car2 = CarAgent("car2", file_name="./dl_sec5.py")
car2.set_initial([[20, -0.5, 0, 1], [20.5, 0.5, 0, 1]], (AgentMode.Normal, TrackMode.T0))
scenario.add_agent(car1)
scenario.add_agent(car2)

traces_veri = scenario.verify(20, 0.01)

import plotly.graph_objects as go
from verse.plotter.plotter2D import *

fig = go.Figure()
fig = reachtube_tree(traces_veri, Map2Lanes(), fig, 1, 2, [1, 2], "lines", "trace")
#fig.show()
fig.write_html('figure.html', auto_open=True)