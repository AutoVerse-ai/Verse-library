from dryvr_plus_plus.example.example_agent.origin_agent import thermo_agent
from dryvr_plus_plus.scene_verifier.scenario.scenario import Scenario
from dryvr_plus_plus.example.example_map.simple_map2 import SimpleMap2, SimpleMap3, SimpleMap5, SimpleMap6
from dryvr_plus_plus.plotter.plotter2D import *
from dryvr_plus_plus.example.example_sensor.fake_sensor import FakeSensor2
from dryvr_plus_plus.example.example_sensor.thermo_sensor import ThermoSensor

import plotly.graph_objects as go
from enum import Enum, auto


class ThermoMode(Enum):
    ON = auto()
    OFF = auto()


if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/thermo_controller.py'
    scenario = Scenario()

    car = thermo_agent('test', file_name=input_code_name)
    scenario.add_agent(car)
    tmp_map = SimpleMap3()
    scenario.set_map(tmp_map)
    scenario.set_sensor(ThermoSensor())
    # modify mode list input
    scenario.set_init(
        [
            [[75.0, 0.0, 0.0], [76, 0.0, 0.0]],
        ],
        [
            tuple([ThermoMode.ON]),
        ]
    )
    traces = scenario.verify(3.5, 0.05)
    fig = go.Figure()
    fig = reachtube_tree(traces, tmp_map, fig, 2, 1,
                         'lines', 'trace', print_dim_list=[2, 1])
    fig.show()
