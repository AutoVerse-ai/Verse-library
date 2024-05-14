from verse.map.lane_segment import StraightLane
from verse.map.lane import Lane
from verse.stars.starset import StarSet
import polytope as pc
from verse.sensor.base_sensor_stars import BaseStarSensor

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
    #TODO: is there bad a reason this needed to be added??
    if not (lane_idx, agent_mode_src, agent_mode_dest) in h_dict.keys():
        return None
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


from verse.scenario import Scenario, ScenarioConfig
from verse.analysis import ReachabilityMethod
import numpy as np

scenario = Scenario(ScenarioConfig(parallel=False))
#scenario = Scenario()

scenario.set_map(Map2Lanes())

from tutorial_agent import CarAgent

car1 = CarAgent("car1", file_name="./dl_sec5_star.py")
initial_set_polytope = pc.box2poly([[0,0.5], [-0.5,0.5], [0,0], [2,2]])
initial_star = StarSet.from_polytope(initial_set_polytope)
#initial_star.basis[0][0] = .5
#initial_star.basis[0][1] = -.85
#initial_star.basis[1][1] = 0.5
#initial_star.basis[1][0] = 0.85

#initial_star.basis[2][0] = .5
#initial_star.basis[2][1] = -.85
#initial_star.basis[3][1] = 0.5
#initial_star.basis[3][0] = 0.85
#initial_star.intersection_halfspace(np.array([1,0,0,0]), 0.2 )
#initial_star.intersection_halfspace(np.array([0,1,0,0]), 0.2 )


car1.set_initial(initial_star, (AgentMode.Normal, TrackMode.T0))
car2 = CarAgent("car2", file_name="./dl_sec5_star.py")
#initial_set_polytope_car2 = pc.box2poly([[20,20.5], [-0.5,0.5], [0,0], [1,1]])
initial_set_polytope_car2 = pc.box2poly([[20,20.5], [-0.5,0.5], [0,0], [1,1]])


car2.set_initial(StarSet.from_polytope(initial_set_polytope_car2), (AgentMode.Normal, TrackMode.T0))
scenario.add_agent(car1)
scenario.add_agent(car2)

scenario.set_sensor(BaseStarSensor())

scenario.config.reachability_method = ReachabilityMethod.STAR_SETS

traces_veri = scenario.verify(8, 0.01)
#traces_veri = scenario.verify(20, 0.01)


import plotly.graph_objects as go
from verse.plotter.plotterStar import *

plot_reachtube_stars(traces_veri, Map2Lanes(), 0 , 1, 10)

#fig = go.Figure()
#fig = reachtube_tree(traces_veri, Map2Lanes(), fig, 1, 2, [1, 2], "lines", "trace")
#fig.show()
#fig.write_html('test_fig.html', auto_open=True)