from verse.scenario import Scenario
from tutorial_map import M4
scenario = Scenario()
scenario.set_map(M4())

from enum import Enum, auto

class CraftMode(Enum):
    Normal = auto()
    MoveUp = auto()
    MoveDown = auto()

class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M10 = auto()
    M12 = auto()
    M21 = auto()

class State:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    craft_mode: CraftMode
    track_mode: TrackMode

    def __init__(self, x, y, z, vx, vy, vz, craft_mode, track_mode):
        pass

from tutorial_agent import DroneAgent
drone1 = DroneAgent(
    'drone1', file_name="dl_sec2.py", t_v_pair=(1, 1), box_side=[0.4]*3)
drone1.set_initial(
    [[1.5, -0.5, -0.5, 0, 0, 0], [2.5, 0.5, 0.5, 0, 0, 0]],
    (CraftMode.Normal, TrackMode.T1)
)
scenario.add_agent(drone1)

drone2 = DroneAgent(
    'drone2', file_name="dl_sec2.py", t_v_pair=(1, 0.5), box_side=[0.4]*3)
drone2.set_initial(
    [[19.5, -0.5, -0.5, 0, 0, 0], [20.5, 0.5, 0.5, 0, 0, 0]],
    (CraftMode.Normal, TrackMode.T1)
)
scenario.add_agent(drone2)

scenario.add_agent(drone1)
scenario.add_agent(drone2)

from tutorial_sensor import DefaultSensor
scenario.set_sensor(DefaultSensor())

traces_simu = scenario.simulate(60, 0.2)
traces_veri = scenario.verify(60, 0.2)

from verse.plotter.plotter3D import *
import pyvista as pv
import warnings
warnings.filterwarnings("ignore")

fig = pv.Plotter()
fig = plot3dMap(M4(), ax=fig)
fig = plot3dReachtube(traces_veri, 'drone1', 1, 2, 3, color = 'r', ax=fig)
fig = plot3dReachtube(traces_veri, 'drone2', 1, 2, 3, color = 'b', ax=fig)
fig.set_background('#e0e0e0')
fig.show()