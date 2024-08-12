
from typing import Tuple, List 

import numpy as np 
from scipy.integrate import ode 

from verse import BaseAgent, Scenario
from verse.scenario import ScenarioConfig
from verse.analysis.analysis_tree import TraceType, AnalysisTree 
from verse.parser import ControllerIR
from verse.analysis import AnalysisTreeNode, AnalysisTree
import copy 

from enum import Enum, auto

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go


#starset
from verse.analysis import ReachabilityMethod
from verse.stars.starset import StarSet
import polytope as pc
from verse.sensor.base_sensor_stars import *
from new_files.star_diams import *

import time

class ThermoAgent(BaseAgent):
    def __init__(
        self, 
        id, 
        code = None,
        file_name = None
    ):
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic_heat(t, state):
        x = state
        x_dot = 40-0.5*x
        return [x_dot]
    
    @staticmethod
    def dynamic_cool(t, state):
        x = state
        x_dot = 30-0.5*x
        return [x_dot]
    
    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            if mode[0]=="Heat":
                r = ode(self.dynamic_heat)
            elif mode[0]=="Cool":
                r = ode(self.dynamic_cool)
            else:
                raise ValueError
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class ThermoMode(Enum):
    Heat=auto()
    Cool=auto()

class State:
    x: float
    agent_mode: ThermoMode 

    def __init__(self, x, agent_mode: ThermoMode):
        pass 

def decisionLogic(ego: State, other: State):
    output = copy.deepcopy(ego)

    if ego.agent_mode == ThermoMode.Heat and ego.x>=75:
        output.agent_mode = ThermoMode.Cool
    if ego.agent_mode == ThermoMode.Cool and ego.x<=65:
        output.agent_mode = ThermoMode.Heat

    return output 



if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "lin_thermo.py")
    Thermo = ThermoAgent('thermo', file_name=input_code_name)

    init_bruss = [[68, 69]]
    initial_set_polytope = pc.box2poly(init_bruss)
    Thermo.set_initial(StarSet.from_polytope(initial_set_polytope),  tuple([ThermoMode.Heat]))
    
    scenario = Scenario()

    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.add_agent(Thermo)

    scenario.set_sensor(BaseStarSensor())

    # init_bruss = [[68], [69]] # setting initial upper bound to 72 causes hyperrectangle to become large fairly quickly
    # # # -----------------------------------------

    # scenario.set_init_single(
    #     'thermo', init_bruss, (ThermoMode.Heat,)
    # )

    ### t=10 takes quite a long time to run, try t=4 like in c2e2 example
    ### seems to actually loop at t=4.14, not sure what that is about -- from first glance, reason seems to be hyperrectangles blowing up in size
    #trace = scenario.verify(4, 0.01)

    start_time = time.time()
    trace = scenario.verify(4, 0.01)
    run_time = time.time() - start_time

    print("time")
    print(run_time)

    # pp_fix(reach_at_fix(trace, 0, 10))

    #print(f'Fixed points exists? {fixed_points_fix(trace, 4, 0.01)}')

    print("diams")
    diams = time_step_diameter(trace, 4, 0.01)
    #print(diams)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    #plot_reachtube_stars(trace, None, 0, 1)


    # fig = go.Figure()
    # fig = reachtube_tree(trace, None, fig, 0, 1, [0, 1], "fill", "trace") 
    # # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    # fig.show()