from verse import Scenario, ScenarioConfig
from mp0 import TrafficSignalAgent, TrafficSensor

from traffic_controller import TLMode 

from verse.plotter.plotter2D import simulation_tree, reachtube_tree 
import plotly.graph_objects as go 

if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "traffic_controller.py")
    traffic_light = TrafficSignalAgent('tl', file_name = input_code_name)
    
    scenario = Scenario(ScenarioConfig(init_seg_length = 1, parallel = False))

    scenario.add_agent(traffic_light)

    scenario.set_sensor(TrafficSensor())

    init_tl = [[140,0,0,0,0],[140,0,0,0,0]]

    scenario.set_init_single('tl', init_tl, (TLMode.GREEN,))

    res = scenario.verify(50,0.1)

    fig = go.Figure()
    fig = reachtube_tree(res, None, fig, 0, 5)
    fig.show()


