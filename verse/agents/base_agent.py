from verse.parser.parser import ControllerIR
import numpy as np 
from scipy.integrate import ode
import copy

class BaseAgent:
    """
        Agent Base class

        Methods
        -------
        TC_simulate
    """
    def __init__(self, id, code = None, file_name = None, initial_state = None, initial_mode = None, static_param = None, uncertain_param = None): 
        """
            Constructor of agent base class.

            Parameters
            ----------
            id : str
                id of the agent.
            code: str
                actual code of python controller
            file_name: str 
                file name to the python controller
        """
        self.controller: ControllerIR = ControllerIR.parse(code, file_name)
        self.id = id
        self.init_cont = copy.deepcopy(initial_state)
        self.init_disc = copy.deepcopy(initial_mode)
        self.static_parameters = copy.deepcopy(static_param)
        self.uncertain_parameters = copy.deepcopy(uncertain_param)

    def set_initial(self, initial_state, initial_mode, static_param = None, uncertain_param = None):
        self.set_initial_state(initial_state)
        self.set_initial_mode(initial_mode)
        self.set_static_parameter(static_param)
        self.set_uncertain_parameter(uncertain_param)

    def set_initial_state(self, initial_state):
        self.init_cont = copy.deepcopy(initial_state) 
    
    def set_initial_mode(self, initial_mode):
        self.init_disc = copy.deepcopy(initial_mode)

    def set_static_parameter(self, static_param):
        self.static_parameters = copy.deepcopy(static_param)

    def set_uncertain_parameter(self, uncertain_param):
        self.uncertain_parameters = copy.deepcopy(uncertain_param)

    def TC_simulate(self, mode, initialSet, time_horizon, time_step, map=None):
        """
        Abstract simulation function

        Parameters
        ----------
            mode: str
                The current mode to simulate
            initialSet: List[float]
                The initial condition to perform the simulation
            time_horizon: float
                The time horizon for simulation
            time_step: float
                time_step for performing simulation
            map: LaneMap, optional
                Provided if the map is used 
        """
        time_bound = float(time_horizon)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        # note: digit of time
        init = initialSet
        trace = [[0]+init]
        for i in range(len(t)):
            r = ode(self.dynamics)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init)
        return np.array(trace)