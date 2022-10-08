from quadrotor_agent import QuadrotorAgent
from verse import Scenario
from verse.plotter.plotter3D_new import *
from verse.plotter.plotter3D import *
from verse.map.example_map.simple_map_3d import SimpleMap6
import pyvista as pv
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


if __name__ == "__main__":
    input_code_name = './demo/tacas2023/exp1/quadrotor_controller3.py'
    input_code_name2 = './demo/tacas2023/exp1/quadrotor_controller4.py'

    scenario = Scenario()
    time_step = 0.1
    quadrotor1 = QuadrotorAgent(
        'test1', file_name=input_code_name, t_v_pair=(1, 1), box_side=[0.4]*3)
    init_l_1 = [9.5, 0, -0.35, 0, 0, 0]
    init_u_1 = [10.2, 0.7, 0.35, 0, 0, 0]
    quadrotor1.set_initial(
        [init_l_1, init_u_1],
        (CraftMode.Normal, TrackMode.T1)
    )
    scenario.add_agent(quadrotor1)

    quadrotor2 = QuadrotorAgent(
        'test2', file_name=input_code_name2, t_v_pair=(1, 0.3), box_side=[0.4]*3)
    init_l_2 = [3, 9, -0.35, 0, 0, 0]
    init_u_2 = [3.7, 9.7, 0.35, 0, 0, 0]
    quadrotor2.set_initial(
        [init_l_2, init_u_2],
        (CraftMode.Normal, TrackMode.T1)
    )
    scenario.add_agent(quadrotor2)

    tmp_map = SimpleMap6()
    scenario.set_map(tmp_map)
    # scenario.set_sensor(QuadrotorSensor())

    # traces = scenario.simulate(40, time_step, seed=4)
    # traces.dump("./output1.json")
    traces = AnalysisTree.load('./output1.json')
    # fig = pv.Plotter()
    # fig = plot3dMap(tmp_map, ax=fig, width=0.05)
    # fig = plot3dReachtube(traces, 'test1',1,2,3,'r',fig, edge = True)
    # fig = plot3dReachtube(traces, 'test2',1,2,3,'g',fig, edge = True)
    # fig.set_background('#e0e0e0')
    # fig.show()
    fig = go.Figure()
    fig = simulation_tree_3d(traces, tmp_map, fig, 1, 2, 3, [1, 2, 3])
    fig.show()
