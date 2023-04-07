from origin_agent import adv_thermo_agent
from verse import Scenario
from verse.parser.parser import ControllerIR
from verse.plotter.plotter2D import *
from verse.scenario.scenario import ScenarioConfig
from verse.sensor.example_sensor.thermo_sensor import ThermoSensor
import plotly.graph_objects as go
from enum import Enum, auto

class ThermoMode(Enum):
    WARM = auto()
    WARM_FAST = auto()
    COOL = auto()
    COOL_FAST = auto()

if __name__ == "__main__":
    input_code_name = './demo/dryvr_demo/adv_thermo_controller.py'
    config = ScenarioConfig()
    scenario = Scenario(config)

    scenario.add_agent(adv_thermo_agent('test', file_name=input_code_name))
    scenario.add_agent(adv_thermo_agent('test2', file_name=input_code_name))
    # for path in scenario.agent_dict["test"].decision_logic.paths:
    #     print(ControllerIR.dump(path.cond_veri))
    #     print(ControllerIR.dump(path.val_veri))
    scenario.set_sensor(ThermoSensor())
    # modify mode list input
    scenario.set_init(
        [
            [[75.0, 0.0, 0.0], [75.0, 0.0, 0.0]],
            [[76.0, 0.0, 0.0], [76.0, 0.0, 0.0]],
        ],
        [
            tuple([ThermoMode.WARM]),
            tuple([ThermoMode.COOL]),
        ]
    )
    traces = scenario.verify(3.5, 0.05)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 2, 1, [2, 1],
                         'lines', 'trace')
    fig.show()
