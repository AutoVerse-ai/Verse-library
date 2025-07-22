from TRAFF_22_agent import CarAgent

from verse.scenario import Scenario, ScenarioConfig
from verse.sensor import BaseSensor
from verse.plotter.plotter2D import *
import plotly.io as pio
import numpy as np


if __name__ == "__main__":
    pio.renderers.default = 'browser'
    scenario = Scenario(ScenarioConfig(parallel=False))

    initial_state = [[1000-0.0004, 2000-0.0004, np.pi-0.006, -0.002, 100-0.002], [1000+0.0004, 2000 + 0.0004, np.pi + 0.006, 0.002, 100+ 0.002]]

    car = CarAgent("car")
    

    scenario.add_agent(car)
    scenario.set_init_single('car', initial_state, init_mode=())


    trace = scenario.simulate(20, 0.005)
    print(trace.root.trace)
    fig = simulation_tree(trace, x_dim=1, y_dim=2)
    fig.show()
